"""Async API runner and incremental JSONL collector."""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .games import Game, distribution_support_for_row
from .parser import parse_action, parse_analytical_correctness
from .prompts import generate_prompt


@dataclass(frozen=True)
class ExperimentJob:
    game: Game
    condition: str
    model_label: str
    model_name: str
    recommendation: str
    trial: int

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.game.name,
            self.condition,
            self.model_label,
            self.recommendation,
            self.trial,
        )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _record_path(results_dir: Path, model_label: str) -> Path:
    return results_dir / f"raw_{_slug(model_label)}.jsonl"


def load_existing_keys(results_dir: Path) -> set[tuple[str, str, str, str, int]]:
    keys = set()
    for path in results_dir.glob("raw_*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.add(
                    (
                        record.get("game"),
                        record.get("condition"),
                        record.get("model"),
                        record.get("recommendation"),
                        int(record.get("trial", -1)),
                    )
                )
    return keys


def build_jobs(
    games: list[Game],
    model_labels: list[str],
    n: int,
    *,
    include_c7_fake: bool = True,
) -> list[ExperimentJob]:
    aliases = config.MODEL_ALIASES
    jobs: list[ExperimentJob] = []
    for game in games:
        real_recs = distribution_support_for_row(game.real_ce)
        fake_recs = distribution_support_for_row(game.fake_ce)
        condition_recs: dict[str, tuple[str, ...]] = {
            "C1": real_recs,
            "C2": real_recs,
            "C3": real_recs,
            "C4": fake_recs,
            "C5": fake_recs,
            "C6-real": ("analysis",),
            "C6-fake": ("analysis",),
            "C7-real": real_recs,
        }
        if include_c7_fake:
            condition_recs["C7-fake"] = fake_recs

        for model_label in model_labels:
            model_name = aliases.get(model_label, model_label)
            for condition, recommendations in condition_recs.items():
                for recommendation in recommendations:
                    for trial in range(n):
                        jobs.append(
                            ExperimentJob(
                                game=game,
                                condition=condition,
                                model_label=model_label,
                                model_name=model_name,
                                recommendation=recommendation,
                                trial=trial,
                            )
                        )
    return jobs


class Runner:
    def __init__(
        self,
        *,
        results_dir: Path,
        concurrency: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        request_timeout: float,
        dry_run: bool = False,
    ) -> None:
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.dry_run = dry_run
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, model_label: str) -> asyncio.Lock:
        if model_label not in self._locks:
            self._locks[model_label] = asyncio.Lock()
        return self._locks[model_label]

    async def _append_record(self, record: dict[str, Any]) -> None:
        path = _record_path(self.results_dir, record["model"])
        async with self._lock_for(record["model"]):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _call_chat(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        max_attempts: int = 6,
    ) -> str:
        if self.dry_run:
            await asyncio.sleep(0)
            return "DRY_RUN\nNo API call was made."

        from openai import (
            APIConnectionError,
            APIError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )

        retryable = (RateLimitError, APIStatusError, APIConnectionError, APIError)
        for attempt in range(max_attempts):
            try:
                async with self.semaphore:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_tokens=max_tokens or self.max_tokens,
                            timeout=self.request_timeout,
                        ),
                        timeout=self.request_timeout + 5,
                    )
                return response.choices[0].message.content or ""
            except asyncio.TimeoutError:
                raise TimeoutError(f"API request exceeded {self.request_timeout} seconds")
            except APITimeoutError:
                raise
            except APIStatusError as exc:
                if exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                    raise
                if attempt == max_attempts - 1:
                    raise
                delay = min(60.0, (2**attempt) + random.random())
                await asyncio.sleep(delay)
            except retryable:
                if attempt == max_attempts - 1:
                    raise
                delay = min(60.0, (2**attempt) + random.random())
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable retry loop exit.")

    async def run_job(self, client: Any, job: ExperimentJob) -> None:
        condition = job.condition
        expected_is_ce = condition.endswith("real")
        timestamp = datetime.now(timezone.utc).isoformat()

        if condition.startswith("C6"):
            prompt = generate_prompt(job.game, condition)
            raw = await self._call_chat(
                client,
                model=job.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            correct, identified = parse_analytical_correctness(raw, expected_is_ce)
            record = {
                "game": job.game.name,
                "condition": condition,
                "model": job.model_label,
                "api_model": job.model_name,
                "recommendation": job.recommendation,
                "trial": job.trial,
                "timestamp": timestamp,
                "raw_response": raw,
                "parsed_action": None,
                "parse_success": None,
                "complied": None,
                "analytical_correct": correct,
                "analytical_identified": identified,
            }
            await self._append_record(record)
            return

        if condition.startswith("C7"):
            messages = generate_prompt(job.game, condition, job.recommendation)
            if not isinstance(messages, list):
                raise TypeError("C7 prompt must be a message list.")
            analysis_raw = await self._call_chat(
                client,
                model=job.model_name,
                messages=[messages[0]],
            )
            messages[1]["content"] = analysis_raw
            raw = await self._call_chat(
                client,
                model=job.model_name,
                messages=messages,
                max_tokens=min(self.max_tokens, 512),
            )
            correct, identified = parse_analytical_correctness(analysis_raw, expected_is_ce)
        else:
            prompt = generate_prompt(job.game, condition, job.recommendation)
            if not isinstance(prompt, str):
                raise TypeError("C1-C5 prompt must be a string.")
            raw = await self._call_chat(
                client,
                model=job.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis_raw = None
            correct = None
            identified = None

        parsed = parse_action(raw, job.game.row_actions)
        record = {
            "game": job.game.name,
            "condition": condition,
            "model": job.model_label,
            "api_model": job.model_name,
            "recommendation": job.recommendation,
            "trial": job.trial,
            "timestamp": timestamp,
            "raw_response": raw,
            "analysis_response": analysis_raw,
            "parsed_action": parsed.parsed_action,
            "parse_success": parsed.parse_success,
            "complied": parsed.parsed_action == job.recommendation
            if parsed.parse_success
            else False,
            "analytical_correct": correct,
            "analytical_identified": identified,
        }
        await self._append_record(record)

    async def run(self, jobs: list[ExperimentJob]) -> None:
        if self.dry_run:
            client = None
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=config.get_api_key(),
                base_url=config.API_BASE_URL,
                timeout=self.request_timeout,
            )

        completed = 0
        total = len(jobs)

        async def one(job: ExperimentJob) -> None:
            nonlocal completed
            try:
                await self.run_job(client, job)
            except Exception as exc:
                record = {
                    "game": job.game.name,
                    "condition": job.condition,
                    "model": job.model_label,
                    "api_model": job.model_name,
                    "recommendation": job.recommendation,
                    "trial": job.trial,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_response": "",
                    "parsed_action": None,
                    "parse_success": False,
                    "complied": False,
                    "analytical_correct": None,
                    "analytical_identified": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
                await self._append_record(record)
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == total:
                print(f"Completed {completed}/{total} jobs", flush=True)

        queue: asyncio.Queue[ExperimentJob | None] = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)

        worker_count = max(1, self.concurrency)

        async def worker() -> None:
            while True:
                job = await queue.get()
                try:
                    if job is None:
                        return
                    await one(job)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        for _ in workers:
            queue.put_nowait(None)
        await queue.join()
        await asyncio.gather(*workers)


async def run_experiments(
    *,
    games: list[Game],
    model_labels: list[str],
    n: int,
    results_dir: Path,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: float,
    dry_run: bool = False,
    include_c7_fake: bool = True,
) -> int:
    jobs = build_jobs(games, model_labels, n, include_c7_fake=include_c7_fake)
    existing = load_existing_keys(results_dir)
    pending = [job for job in jobs if job.key not in existing]
    print(
        f"Total jobs: {len(jobs)}; existing: {len(jobs) - len(pending)}; pending: {len(pending)}",
        flush=True,
    )
    if not pending:
        return 0
    runner = Runner(
        results_dir=results_dir,
        concurrency=concurrency,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        dry_run=dry_run,
    )
    await runner.run(pending)
    return len(pending)
