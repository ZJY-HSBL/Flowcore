import torch

from flowcore import FlowConfig, FlowCoreModel, RefinedFlowCoreModel


def _config(refinement_steps=1):
    return FlowConfig(
        input_dim=5,
        output_dim=3,
        state_dim=32,
        block_size=4,
        num_ports=4,
        context_dim=2,
        controller_hidden_dim=24,
        encoder_hidden_dim=16,
        readout_hidden_dim=16,
        active_ports=2,
        refinement_steps=refinement_steps,
    )


def test_model_parallel_matches_sequential():
    torch.manual_seed(5)
    model = FlowCoreModel(_config())
    x = torch.randn(3, 23, 5)
    context = torch.randn(3, 2)
    par = model(x, context=context, mode="parallel")
    seq = model(x, context=context, mode="sequential")
    torch.testing.assert_close(par.states, seq.states, rtol=4e-5, atol=4e-5)
    torch.testing.assert_close(par.output, seq.output, rtol=4e-5, atol=4e-5)


def test_refined_model_shapes_and_backward():
    torch.manual_seed(6)
    model = RefinedFlowCoreModel(_config(refinement_steps=2), state_summary_dim=8)
    x = torch.randn(2, 13, 5)
    out = model(x, context=torch.randn(2, 2), mode="parallel")
    assert out.output.shape == (2, 13, 3)
    assert out.states.shape == (2, 13, 32)
    loss = out.output.square().mean()
    loss.backward()
    assert model.substrate.raw_blocks.grad is not None
    assert model.controller.route_head.weight.grad is not None
