"""StepPlayer managing playback cursor and navigation.

Single responsibility: navigate backward, forward, and jump across playlist steps.
Allowed dependencies: tilevis.core, tilevis.render, stdlib.
"""

from __future__ import annotations

from tilevis.core.config import PresenterSettings
from tilevis.core.trace import Trace
from tilevis.render.playlist import StepRef, build_playlist


class StepPlayer:
    """Bounded cursor supporting forward/backward, stage/pass/block jumps, and re-anchoring."""

    def __init__(self, trace: Trace, settings: PresenterSettings) -> None:
        self.trace = trace
        self.settings = settings
        self.playlist: list[StepRef] = build_playlist(trace, settings)
        self.cursor: int = 0

    @property
    def total_steps(self) -> int:
        """Total number of steps in the active playlist."""
        return len(self.playlist)

    @property
    def current(self) -> StepRef | None:
        """Currently active StepRef, or None if playlist is empty."""
        if not self.playlist:
            return None
        return self.playlist[self.cursor]

    @property
    def can_next(self) -> bool:
        """True if advancing one step is possible."""
        return bool(self.playlist) and self.cursor + 1 < len(self.playlist)

    @property
    def can_prev(self) -> bool:
        """True if stepping backward is possible."""
        return bool(self.playlist) and self.cursor > 0

    def next(self) -> bool:
        """Advance one step forward (bounded; does not wrap)."""
        if not self.can_next:
            return False
        self.cursor += 1
        return True

    def prev(self) -> bool:
        """Move one step backward (bounded; does not wrap)."""
        if not self.can_prev:
            return False
        self.cursor -= 1
        return True

    def first(self) -> bool:
        """Jump to the first step in the playlist."""
        if not self.playlist or self.cursor == 0:
            return False
        self.cursor = 0
        return True

    def last(self) -> bool:
        """Jump to the last step in the playlist."""
        if not self.playlist or self.cursor == len(self.playlist) - 1:
            return False
        self.cursor = len(self.playlist) - 1
        return True

    def next_stage(self) -> bool:
        """Jump to the first step of the next (pass_index, stage) boundary."""
        if not self.playlist or self.cursor >= len(self.playlist) - 1:
            return False
        cur = self.playlist[self.cursor]
        cur_key = (cur.pass_index, cur.stage)
        for idx in range(self.cursor + 1, len(self.playlist)):
            cand = self.playlist[idx]
            if (cand.pass_index, cand.stage) != cur_key:
                self.cursor = idx
                return True
        return False

    def prev_stage(self) -> bool:
        """Jump to the start of the previous (pass_index, stage) group."""
        if not self.playlist or self.cursor <= 0:
            return False
        cur = self.playlist[self.cursor]
        cur_key = (cur.pass_index, cur.stage)

        # Find start index of the current (pass_index, stage) group
        cur_start = self.cursor
        while (
            cur_start > 0
            and (
                self.playlist[cur_start - 1].pass_index,
                self.playlist[cur_start - 1].stage,
            )
            == cur_key
        ):
            cur_start -= 1

        if cur_start == 0:
            if self.cursor > 0:
                self.cursor = 0
                return True
            return False

        prev_idx = cur_start - 1
        prev_key = (self.playlist[prev_idx].pass_index, self.playlist[prev_idx].stage)
        while (
            prev_idx > 0
            and (
                self.playlist[prev_idx - 1].pass_index,
                self.playlist[prev_idx - 1].stage,
            )
            == prev_key
        ):
            prev_idx -= 1

        self.cursor = prev_idx
        return True

    def next_pass(self) -> bool:
        """Jump to the first step of the next forward pass (no wrap-around)."""
        if not self.playlist:
            return False
        cur = self.playlist[self.cursor]
        if cur.pass_index is None:
            if self.cursor > 0:
                # On END card -> cannot advance to a later pass
                return False
            for idx, ref in enumerate(self.playlist):
                if ref.pass_index == 0:
                    self.cursor = idx
                    return True
            return False

        target_pass = cur.pass_index + 1
        for idx in range(self.cursor + 1, len(self.playlist)):
            ref = self.playlist[idx]
            if ref.pass_index is not None and ref.pass_index >= target_pass:
                self.cursor = idx
                return True
        return False

    def goto(self, ref: StepRef | str) -> bool:
        """Jump directly to a matching StepRef or step_id string."""
        if not self.playlist:
            return False
        target_id = ref.step_id if isinstance(ref, StepRef) else str(ref)
        target_pass = ref.pass_index if isinstance(ref, StepRef) else None

        for idx, cand in enumerate(self.playlist):
            if cand.step_id == target_id and (
                target_pass is None or cand.pass_index == target_pass
            ):
                if idx == self.cursor:
                    return False
                self.cursor = idx
                return True
        return False

    def goto_block(self, pass_index: int, block_index: int) -> bool:
        """Jump to the first visible step matching (pass_index, block_index)."""
        if not self.playlist:
            return False
        for idx, cand in enumerate(self.playlist):
            if cand.pass_index == pass_index and cand.block_index == block_index:
                if idx == self.cursor:
                    return False
                self.cursor = idx
                return True
        return False

    def update_settings(self, new_settings: PresenterSettings) -> StepRef | None:
        """Rebuild playlist for new_settings and re-anchor cursor to the nearest step."""
        old_ref = self.current
        old_cursor = self.cursor
        self.settings = new_settings
        self.playlist = build_playlist(self.trace, new_settings)

        if not self.playlist:
            self.cursor = 0
            return None

        if old_ref is None:
            self.cursor = min(old_cursor, len(self.playlist) - 1)
            return self.current

        # 1. Exact step_id + pass_index match
        for idx, cand in enumerate(self.playlist):
            if cand.step_id == old_ref.step_id and cand.pass_index == old_ref.pass_index:
                self.cursor = idx
                return self.current

        # 2. Same pass_index and block_index (closest detail level)
        if old_ref.pass_index is not None and old_ref.block_index is not None:
            matches = [
                (abs(cand.detail_level - old_ref.detail_level), idx)
                for idx, cand in enumerate(self.playlist)
                if cand.pass_index == old_ref.pass_index and cand.block_index == old_ref.block_index
            ]
            if matches:
                matches.sort(key=lambda item: (item[0], item[1]))
                self.cursor = matches[0][1]
                return self.current

        # 3. Same pass_index and stage
        if old_ref.pass_index is not None:
            for idx, cand in enumerate(self.playlist):
                if cand.pass_index == old_ref.pass_index and cand.stage == old_ref.stage:
                    self.cursor = idx
                    return self.current

            # 4. Closest pass_index
            for idx, cand in enumerate(self.playlist):
                if cand.pass_index == old_ref.pass_index:
                    self.cursor = idx
                    return self.current

        # 5. Bounded fallback
        self.cursor = min(old_cursor, len(self.playlist) - 1)
        return self.current
