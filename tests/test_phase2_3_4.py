# test_phase2_3_4.py
"""
Tests for Phase 2, 3, and 4 implementations.

This file tests the Cython-optimized implementations of:
- CNN layers (Conv2d, MaxPool2d, AvgPool2d)
- RNN layers (RNNCell, LSTMCell, GRUCell)
- Advanced features (attention, transformer, GNN)
- Optimizers (SGD, Adam)
- Model utilities (Sequential, save/load)
"""

import numpy as np


# ============================================================================
# Phase 2: CNN Tests
# ============================================================================

def test_conv2d_forward():
    """Test Conv2d forward pass."""
    from miniten.nn.cnn_impl import Conv2d
    
    # Create Conv2d layer: 3 input channels, 16 output channels, 3x3 kernel
    conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    
    # Input: batch=2, channels=3, height=8, width=8
    x = np.random.randn(2, 3, 8, 8).astype(np.float64)
    
    # Forward pass
    output = conv.forward(x)
    
    # Check output shape
    assert output.shape == (2, 16, 8, 8), f"Expected (2, 16, 8, 8), got {output.shape}"
    print("Conv2d forward pass: OK")


def test_conv2d_backward():
    """Test Conv2d backward pass."""
    from miniten.nn.cnn_impl import Conv2d
    
    conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(2, 3, 8, 8).astype(np.float64)
    
    # Forward
    output = conv.forward(x)
    
    # Backward
    grad_output = np.random.randn(*output.shape).astype(np.float64)
    grad_input = conv.backward(grad_output)
    
    # Check gradient shape
    assert grad_input.shape == x.shape, f"Expected {x.shape}, got {grad_input.shape}"
    assert conv.grad_weight.shape == conv.weight.shape
    print("Conv2d backward pass: OK")


def test_maxpool2d():
    """Test MaxPool2d forward and backward."""
    from miniten.nn.cnn_impl import maxpool2d_forward, maxpool2d_backward
    
    x = np.random.randn(2, 4, 8, 8).astype(np.float64)
    
    # Forward
    output, indices = maxpool2d_forward(x, kernel_size=2, stride=2)
    
    assert output.shape == (2, 4, 4, 4), f"Expected (2, 4, 4, 4), got {output.shape}"
    
    # Backward
    grad_output = np.random.randn(*output.shape).astype(np.float64)
    grad_input = maxpool2d_backward(grad_output, indices, x.shape, kernel_size=2, stride=2)
    
    assert grad_input.shape == x.shape
    print("MaxPool2d: OK")


def test_avgpool2d():
    """Test AvgPool2d forward and backward."""
    from miniten.nn.cnn_impl import avgpool2d_forward, avgpool2d_backward
    
    x = np.random.randn(2, 4, 8, 8).astype(np.float64)
    
    # Forward
    output, _ = avgpool2d_forward(x, kernel_size=2, stride=2)
    
    assert output.shape == (2, 4, 4, 4), f"Expected (2, 4, 4, 4), got {output.shape}"
    
    # Backward
    grad_output = np.random.randn(*output.shape).astype(np.float64)
    grad_input = avgpool2d_backward(grad_output, x.shape, kernel_size=2, stride=2)
    
    assert grad_input.shape == x.shape
    print("AvgPool2d: OK")


def test_dropout():
    """Test dropout forward and backward."""
    from miniten.nn.cnn_impl import dropout_forward, dropout_backward
    
    x = np.random.randn(16, 64).astype(np.float64)
    
    # Training mode
    output, mask = dropout_forward(x, p=0.5, training=True)
    assert output.shape == x.shape
    assert mask is not None
    
    # Eval mode
    output_eval, mask_eval = dropout_forward(x, p=0.5, training=False)
    assert mask_eval is None
    np.testing.assert_array_equal(output_eval, x)
    
    print("Dropout: OK")


# ============================================================================
# Phase 2: RNN Tests
# ============================================================================

def test_rnn_cell():
    """Test RNNCell forward and backward."""
    from miniten.nn.rnn_impl import RNNCell
    
    cell = RNNCell(input_size=10, hidden_size=20)
    
    # Input: batch=4, features=10
    x = np.random.randn(4, 10).astype(np.float64)
    
    # Forward
    h = cell.forward(x)
    
    assert h.shape == (4, 20), f"Expected (4, 20), got {h.shape}"
    
    # Backward
    grad_h = np.random.randn(*h.shape).astype(np.float64)
    grad_x, grad_h_prev = cell.backward(grad_h)
    
    assert grad_x.shape == x.shape
    assert grad_h_prev.shape == h.shape
    print("RNNCell: OK")


def test_lstm_cell():
    """Test LSTMCell forward and backward."""
    from miniten.nn.rnn_impl import LSTMCell
    
    cell = LSTMCell(input_size=10, hidden_size=20)
    
    x = np.random.randn(4, 10).astype(np.float64)
    
    # Forward
    h, c = cell.forward(x)
    
    assert h.shape == (4, 20)
    assert c.shape == (4, 20)
    
    # Backward
    grad_h = np.random.randn(*h.shape).astype(np.float64)
    grad_c = np.random.randn(*c.shape).astype(np.float64)
    grad_x, grad_h_prev, grad_c_prev = cell.backward(grad_h, grad_c)
    
    assert grad_x.shape == x.shape
    print("LSTMCell: OK")


def test_gru_cell():
    """Test GRUCell forward and backward."""
    from miniten.nn.rnn_impl import GRUCell
    
    cell = GRUCell(input_size=10, hidden_size=20)
    
    x = np.random.randn(4, 10).astype(np.float64)
    
    # Forward
    h = cell.forward(x)
    
    assert h.shape == (4, 20)
    
    # Backward
    grad_h = np.random.randn(*h.shape).astype(np.float64)
    grad_x, grad_h_prev = cell.backward(grad_h)
    
    assert grad_x.shape == x.shape
    print("GRUCell: OK")


def test_rnn_sequence():
    """Test RNN sequence processing."""
    from miniten.nn.rnn_impl import RNNCell, rnn_forward_sequence
    
    cell = RNNCell(input_size=10, hidden_size=20)
    
    # Sequence: seq_len=5, batch=4, features=10
    x = np.random.randn(5, 4, 10).astype(np.float64)
    
    outputs, h_final = rnn_forward_sequence(cell, x)
    
    assert outputs.shape == (5, 4, 20)
    assert h_final.shape == (4, 20)
    print("RNN sequence: OK")


def test_lstm_sequence():
    """Test LSTM sequence processing."""
    from miniten.nn.rnn_impl import LSTMCell, lstm_forward_sequence
    
    cell = LSTMCell(input_size=10, hidden_size=20)
    
    x = np.random.randn(5, 4, 10).astype(np.float64)
    
    outputs, (h_final, c_final) = lstm_forward_sequence(cell, x)
    
    assert outputs.shape == (5, 4, 20)
    assert h_final.shape == (4, 20)
    assert c_final.shape == (4, 20)
    print("LSTM sequence: OK")


# ============================================================================
# Phase 3: Advanced Features Tests
# ============================================================================

def test_graph_conv():
    """Test GraphConv layer."""
    from miniten.nn.advanced_impl import GraphConv
    
    gcn = GraphConv(in_features=16, out_features=32)
    
    # 100 nodes, 16 features each
    x = np.random.randn(100, 16).astype(np.float64)
    
    # Random adjacency matrix (symmetric)
    adj = np.random.rand(100, 100)
    adj = (adj + adj.T) / 2  # Symmetrize
    adj = (adj > 0.9).astype(np.float64)  # Sparse
    np.fill_diagonal(adj, 1.0)  # Self-loops
    
    # Forward
    output = gcn.forward(x, adj)
    
    assert output.shape == (100, 32)
    
    # Backward
    grad_output = np.random.randn(*output.shape).astype(np.float64)
    grad_input = gcn.backward(grad_output)
    
    assert grad_input.shape == x.shape
    print("GraphConv: OK")


def test_scaled_dot_product_attention():
    """Test scaled dot-product attention."""
    from miniten.nn.advanced_impl import scaled_dot_product_attention
    
    batch_size, seq_len, d_k = 2, 10, 64
    
    query = np.random.randn(batch_size, seq_len, d_k).astype(np.float64)
    key = np.random.randn(batch_size, seq_len, d_k).astype(np.float64)
    value = np.random.randn(batch_size, seq_len, d_k).astype(np.float64)
    
    output, attn_weights = scaled_dot_product_attention(query, key, value)
    
    assert output.shape == (batch_size, seq_len, d_k)
    assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    # Check attention weights sum to 1
    attn_sums = np.sum(attn_weights, axis=-1)
    np.testing.assert_array_almost_equal(attn_sums, np.ones((batch_size, seq_len)))
    print("Scaled dot-product attention: OK")


def test_multi_head_attention():
    """Test multi-head attention."""
    from miniten.nn.advanced_impl import MultiHeadAttention
    
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    
    batch_size, seq_len = 2, 10
    x = np.random.randn(batch_size, seq_len, 64).astype(np.float64)
    
    output = mha.forward(x, x, x)
    
    assert output.shape == (batch_size, seq_len, 64)
    print("Multi-head attention: OK")


def test_layer_norm():
    """Test layer normalization."""
    from miniten.nn.advanced_impl import layer_norm
    
    batch_size, seq_len, embed_dim = 2, 10, 64
    x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float64)
    gamma = np.ones(embed_dim, dtype=np.float64)
    beta = np.zeros(embed_dim, dtype=np.float64)
    
    output = layer_norm(x, gamma, beta)
    
    assert output.shape == x.shape
    
    # Check normalization (mean should be close to 0, std close to 1)
    for b in range(batch_size):
        for s in range(seq_len):
            mean = np.mean(output[b, s])
            std = np.std(output[b, s])
            assert abs(mean) < 1e-5, f"Mean should be ~0, got {mean}"
            assert abs(std - 1.0) < 1e-5, f"Std should be ~1, got {std}"
    
    print("Layer norm: OK")


def test_transformer_encoder_layer():
    """Test Transformer encoder layer."""
    from miniten.nn.advanced_impl import TransformerEncoderLayer
    
    encoder = TransformerEncoderLayer(embed_dim=64, num_heads=8, ff_dim=256)
    
    batch_size, seq_len = 2, 10
    x = np.random.randn(batch_size, seq_len, 64).astype(np.float64)
    
    output = encoder.forward(x)
    
    assert output.shape == (batch_size, seq_len, 64)
    print("Transformer encoder layer: OK")


def test_quantization():
    """Test model quantization."""
    from miniten.nn.advanced_impl import quantize_tensor_int8, dequantize_tensor_int8
    
    # Create a tensor
    x = np.random.randn(100, 100).astype(np.float64)
    
    # Quantize
    x_quant, scale = quantize_tensor_int8(x)
    
    assert x_quant.dtype == np.int8
    assert x_quant.shape == x.shape
    
    # Dequantize
    x_deq = dequantize_tensor_int8(x_quant, scale)
    
    # Check reconstruction error
    error = np.mean(np.abs(x - x_deq))
    assert error < 0.1, f"Quantization error too high: {error}"
    
    print(f"Quantization: OK (mean error: {error:.6f})")


def test_pruning():
    """Test weight pruning."""
    from miniten.nn.advanced_impl import magnitude_prune, compute_sparsity
    
    # Create weight matrix
    weights = np.random.randn(100, 100).astype(np.float64)
    
    # Prune 50% of weights
    pruned, mask = magnitude_prune(weights, sparsity=0.5)
    
    # Check sparsity
    sparsity = compute_sparsity(pruned)
    assert 0.45 < sparsity < 0.55, f"Expected ~50% sparsity, got {sparsity*100:.1f}%"
    
    print(f"Pruning: OK (sparsity: {sparsity*100:.1f}%)")


def test_positional_encoding():
    """Test sinusoidal positional encoding."""
    from miniten.nn.advanced_impl import sinusoidal_position_encoding
    
    pe = sinusoidal_position_encoding(max_len=100, embed_dim=64)
    
    assert pe.shape == (100, 64)
    
    # Check that position 0 has specific patterns
    assert pe[0, 0] == 0.0  # sin(0) = 0
    
    print("Positional encoding: OK")


# ============================================================================
# Phase 4: Optimizer Tests
# ============================================================================

def test_sgd_update():
    """Test SGD optimizer update."""
    from miniten.optim.optim_impl import sgd_update
    
    param = np.random.randn(10, 10).astype(np.float64)
    param_copy = param.copy()
    grad = np.random.randn(10, 10).astype(np.float64)
    velocity = np.zeros_like(param)
    
    sgd_update(param, grad, velocity, lr=0.1, momentum=0.9, weight_decay=0.0)
    
    # Check parameters changed
    assert not np.allclose(param, param_copy)
    print("SGD update: OK")


def test_adam_update():
    """Test Adam optimizer update."""
    from miniten.optim.optim_impl import adam_update
    
    param = np.random.randn(10, 10).astype(np.float64)
    param_copy = param.copy()
    grad = np.random.randn(10, 10).astype(np.float64)
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    
    adam_update(param, grad, m, v, lr=0.001, beta1=0.9, beta2=0.999,
                eps=1e-8, weight_decay=0.0, t=1)
    
    # Check parameters changed
    assert not np.allclose(param, param_copy)
    print("Adam update: OK")


def test_lr_schedulers():
    """Test learning rate schedulers."""
    from miniten.optim.optim_impl import step_lr, cosine_annealing_lr, warmup_lr
    
    # Step LR: after 2 decay steps (epoch=10, step_size=5), lr = 0.1 * 0.1^2 = 0.001
    lr = step_lr(initial_lr=0.1, epoch=10, step_size=5, gamma=0.1)
    assert abs(lr - 0.001) < 1e-10, f"Expected 0.001, got {lr}"
    
    # Cosine annealing at midpoint should be ~0.05
    lr = cosine_annealing_lr(initial_lr=0.1, epoch=50, T_max=100)
    assert 0.04 < lr < 0.06, f"Expected ~0.05, got {lr}"
    
    # Warmup: at step 5/10, lr = 0.1 * 0.5 = 0.05
    lr = warmup_lr(target_lr=0.1, current_step=5, warmup_steps=10)
    assert abs(lr - 0.05) < 1e-10
    
    print("LR schedulers: OK")


def test_gradient_clipping():
    """Test gradient clipping."""
    from miniten.optim.optim_impl import clip_grad_norm, clip_grad_value
    
    # Test clip by norm
    grads = [np.ones((10, 10), dtype=np.float64) * 2.0]
    total_norm = clip_grad_norm(grads, max_norm=1.0)
    
    new_norm = np.sqrt(np.sum(grads[0] ** 2))
    assert new_norm <= 1.0 + 1e-6
    
    # Test clip by value
    grads = [np.ones((10, 10), dtype=np.float64) * 5.0]
    clip_grad_value(grads, clip_value=1.0)
    
    assert np.all(grads[0] <= 1.0)
    
    print("Gradient clipping: OK")


# ============================================================================
# Phase 4: Deployment Tests
# ============================================================================

def test_sequential():
    """Test Sequential container."""
    from miniten.utils.deployment import Sequential
    from miniten.nn.layers_impl import Linear
    
    # Create a simple network
    model = Sequential([
        Linear(10, 20),
        Linear(20, 5)
    ])
    
    x = np.random.randn(4, 10).astype(np.float64)
    output = model.forward(x)
    
    assert output.shape == (4, 5)
    print("Sequential: OK")


def test_memory_pool():
    """Test memory pool."""
    from miniten.utils.deployment import MemoryPool
    
    pool = MemoryPool(max_size_mb=10)
    
    # Allocate
    arr1 = pool.allocate((100, 100), np.float64)
    arr2 = pool.allocate((100, 100), np.float64)
    
    assert arr1.shape == (100, 100)
    assert arr2.shape == (100, 100)
    
    # Release
    pool.release(arr1)
    
    # Reallocate should reuse
    arr3 = pool.allocate((100, 100), np.float64)
    
    # Clear
    pool.clear()
    
    print("Memory pool: OK")


def test_model_size_estimation():
    """Test model size estimation."""
    from miniten.utils.deployment import Sequential, estimate_model_size
    from miniten.nn.layers_impl import Linear
    
    model = Sequential([
        Linear(784, 256),
        Linear(256, 128),
        Linear(128, 10)
    ])
    
    size_info = estimate_model_size(model)
    
    assert 'total_bytes' in size_info
    assert 'total_mb' in size_info
    assert size_info['total_params'] > 0
    
    print(f"Model size estimation: OK ({size_info['total_params']} params, {size_info['total_mb']:.2f} MB)")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Phase 2, 3, 4 Implementations")
    print("=" * 60)
    print()
    
    print("--- Phase 2: CNN Layers ---")
    test_conv2d_forward()
    test_conv2d_backward()
    test_maxpool2d()
    test_avgpool2d()
    test_dropout()
    print()
    
    print("--- Phase 2: RNN Layers ---")
    test_rnn_cell()
    test_lstm_cell()
    test_gru_cell()
    test_rnn_sequence()
    test_lstm_sequence()
    print()
    
    print("--- Phase 3: Advanced Features ---")
    test_graph_conv()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_layer_norm()
    test_transformer_encoder_layer()
    test_quantization()
    test_pruning()
    test_positional_encoding()
    print()
    
    print("--- Phase 4: Optimizers ---")
    test_sgd_update()
    test_adam_update()
    test_lr_schedulers()
    test_gradient_clipping()
    print()
    
    print("--- Phase 4: Deployment ---")
    test_sequential()
    test_memory_pool()
    test_model_size_estimation()
    print()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
