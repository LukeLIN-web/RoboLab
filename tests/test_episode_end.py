# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A server can end an env's episode early; the env is frozen as truncated."""

import numpy as np
import torch

from robolab.core.environments.env import RobolabEnv
from robolab.eval.base_client import EPISODE_DONE_KEY, InferenceClient


class _Client(InferenceClient):
    """Server ends env 1's episode; env 0 keeps going."""

    open_loop_horizon = 8

    def _extract_observation(self, raw_obs, *, env_id=0):
        return {"env_id": env_id}

    def _pack_request(self, extracted_obs, instruction):
        return extracted_obs

    def _query_server(self, request):
        return {"actions": np.zeros((8, 8)), EPISODE_DONE_KEY: request["env_id"] == 1}

    def _unpack_response(self, response):
        return response["actions"]


def test_server_can_end_an_env_until_reset():
    client = _Client()
    client.infer_batch(None, "task", env_ids=[0, 1])
    assert client.ended_env_ids() == {1}
    client.reset(env_id=1)
    assert client.ended_env_ids() == set()
    client.infer_batch(None, "task", env_ids=[0, 1])
    client.reset()
    assert client.ended_env_ids() == set()


def test_ended_env_is_frozen_as_truncated_at_its_step():
    env = RobolabEnv.__new__(RobolabEnv)  # the bookkeeping only, no scene
    env._frozen_envs = torch.zeros(2, dtype=torch.bool)
    env.episode_length_buf = torch.tensor([10, 20])
    env._env_results, env._env_term_step = {}, {}
    env.recorder_manager = None

    env.end_envs([1])
    env.end_envs([1])  # already frozen: nothing changes

    assert env.active_env_ids == [0]
    assert not env.all_terminated
    assert env.get_env_results()[1] == {"env_id": 1, "success": False, "step": 20}
