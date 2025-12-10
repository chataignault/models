"""Checkpoint utilities for DDPM training with Orbax."""

import os
import time
from typing import Optional, Any
from orbax.checkpoint import (
    CheckpointManager,
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    AsyncCheckpointer,
)


class DDPMCheckpointManager:
    """
    Manages checkpointing for distributed DDPM training.

    Handles saving and restoring model state with async saves for non-blocking checkpointing.
    Automatically unreplicates state before saving to avoid storing multiple copies.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        save_interval_steps: Optional[int] = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to retain
            save_interval_steps: Save every N steps (if None, manual saving only)
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.save_interval_steps = save_interval_steps

        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create async checkpointer for non-blocking saves
        self.checkpointer = AsyncCheckpointer(PyTreeCheckpointer())

        # Configure checkpoint manager options
        options = CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
        )

        # Initialize checkpoint manager
        self.manager = CheckpointManager(
            checkpoint_dir,
            self.checkpointer,
            options=options,
        )

    def save_checkpoint(
        self,
        step: int,
        state: Any,
        is_replicated: bool = True,
        force: bool = False,
    ):
        """
        Save checkpoint asynchronously.

        Args:
            step: Training step number
            state: TrainState (replicated or unreplicated)
            is_replicated: If True, unreplicate before saving
            force: If True, save even if not at save_interval_steps
        """
        # Check if we should save at this step
        if not force and self.save_interval_steps is not None:
            if step % self.save_interval_steps != 0:
                return

        # Unreplicate to save only one copy
        if is_replicated:
            from utils_jax.tpu_utils import unreplicate_first
            state = unreplicate_first(state)

        # Prepare checkpoint data with metadata
        checkpoint_data = {
            'state': state,
            'metadata': {
                'step': step,
                'timestamp': time.time(),
            }
        }

        # Save asynchronously
        self.manager.save(step, checkpoint_data)
        print(f"Checkpoint saved at step {step} to {self.checkpoint_dir}")

    def restore_checkpoint(
        self,
        step: Optional[int] = None,
    ) -> tuple[Optional[Any], Optional[int]]:
        """
        Restore checkpoint from specified step or latest.

        Args:
            step: Step number to restore (None = latest)

        Returns:
            Tuple of (state, step) or (None, None) if no checkpoint found
        """
        # Get step to restore
        if step is None:
            step = self.manager.latest_step()

        # Check if checkpoint exists
        if step is None:
            print("No checkpoint found")
            return None, None

        # Restore checkpoint
        restored = self.manager.restore(step)
        print(f"Checkpoint restored from step {step}")

        return restored['state'], step

    def wait_until_finished(self):
        """Block until all async checkpoint saves complete."""
        self.manager.wait_until_finished()
        print("All checkpoint saves completed")

    def all_steps(self) -> list[int]:
        """
        Get list of all available checkpoint steps.

        Returns:
            List of step numbers with saved checkpoints
        """
        return self.manager.all_steps()

    def latest_step(self) -> Optional[int]:
        """
        Get the latest checkpoint step.

        Returns:
            Latest step number or None if no checkpoints
        """
        return self.manager.latest_step()

    def close(self):
        """Close the checkpoint manager and wait for pending saves."""
        self.wait_until_finished()
        self.manager.close()
