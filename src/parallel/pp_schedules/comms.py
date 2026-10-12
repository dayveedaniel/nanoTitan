import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def recv_forward(microbatch_id: int, prev_pp_rank: int, pp_group: ProcessGroup) -> torch.Tensor:
    # somehow, get shape of buffer, or buffer itself
    logger.debug("Receiving forward microbatch %s", microbatch_id)
    dist.recv(stage_input, src=prev_pp_rank, group=pp_group)
    return stage_input


def send_forward(
    microbatch_id: int, stage_output: torch.Tensor, next_pp_rank: int, pp_group: ProcessGroup
):
    logger.debug("Sending forward microbatch %s", microbatch_id)
    dist.send(stage_output, dst=next_pp_rank, group=pp_group)


def recv_backward(microbatch_id: int, next_pp_rank: int, pp_group: ProcessGroup):
    # somehow, get shape of buffer, or buffer itself
    logger.debug("Receiving backward microbatch %s", microbatch_id)
    dist.recv(output_grad, src=next_pp_rank, group=pp_group)
    return output_grad


def send_backward(
    microbatch_id: int, input_grad: torch.Tensor, prev_pp_rank: int, pp_group: ProcessGroup
):
    logger.debug("Sending backward microbatch %s", microbatch_id)
    dist.send(input_grad, dst=prev_pp_rank, group=pp_group)
