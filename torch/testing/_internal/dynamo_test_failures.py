# tests in this list will run without Dynamo strict mode by default.
FIXME_default_non_strict = {
    "backends/xeon/test_launch",
    "benchmark_utils/test_benchmark_utils",
    "distributions/test_constraints",
    "distributions/test_distributions",
    "dynamo/test_activation_checkpointing",
    "dynamo/test_after_aot",
    "dynamo/test_aot_autograd",
    "dynamo/test_autograd_function",
    "dynamo/test_backends",
    "dynamo/test_backward_higher_order_ops",
    "dynamo/test_base_output",
    "dynamo/test_bytecode_hook",
    "dynamo/test_compile",
    "dynamo/test_comptime",
    "dynamo/test_config",
    "dynamo/test_ctx_manager",
    "dynamo/test_cudagraphs",
    "dynamo/test_debug_utils",
    "dynamo/test_decorators",
    "dynamo/test_dynamic_shapes",
    "dynamo/test_exc",
    "dynamo/test_export",
    "dynamo/test_export_mutations",
    "dynamo/test_frame_init",
    "dynamo/test_functions",
    "dynamo/test_global",
    "dynamo/test_higher_order_ops",
    "dynamo/test_hooks",
    "dynamo/test_input_attr_tracking",
    "dynamo/test_interop",
    "dynamo/test_logging",
    "dynamo/test_minifier",
    "dynamo/test_misc",
    "dynamo/test_model_output",
    "dynamo/test_modules",
    "dynamo/test_nops",
    "dynamo/test_optimizers",
    "dynamo/test_pre_dispatch",
    "dynamo/test_profiler",
    "dynamo/test_python_autograd",
    "dynamo/test_recompile_ux",
    "dynamo/test_recompiles",
    "dynamo/test_replay_record",
    "dynamo/test_repros",
    "dynamo/test_skip_non_tensor",
    "dynamo/test_sources",
    "dynamo/test_subclasses",
    "dynamo/test_subgraphs",
    "dynamo/test_torchrec",
    "dynamo/test_trace_rules",
    "dynamo/test_unspec",
    "dynamo/test_verify_correctness",
    "export/test_db",
    "export/test_experimental",
    "export/test_export",
    "export/test_export_nonstrict",
    "export/test_functionalized_assertions",
    "export/test_pass_infra",
    "export/test_passes",
    "export/test_retraceability",
    "export/test_serdes",
    "export/test_serialize",
    "export/test_unflatten",
    "export/test_upgrade",
    "export/test_verifier",
    "functorch/test_aotdispatch",
    "functorch/test_control_flow",
    "functorch/test_dims",
    "functorch/test_eager_transforms",
    "functorch/test_logging",
    "functorch/test_memory_efficient_fusion",
    "functorch/test_minifier",
    "functorch/test_ops",
    "functorch/test_parsing",
    "functorch/test_rearrange",
    "functorch/test_vmap",
    "functorch/test_vmap_registrations",
    "inductor/test_aot_inductor",
    "inductor/test_aot_inductor_utils",
    "inductor/test_benchmark_fusion",
    "inductor/test_binary_folding",
    "inductor/test_codecache",
    "inductor/test_codegen_triton",
    "inductor/test_compiled_autograd",
    "inductor/test_compiled_optimizers",
    "inductor/test_config",
    "inductor/test_coordinate_descent_tuner",
    "inductor/test_cpu_cpp_wrapper",
    "inductor/test_cpu_repro",
    "inductor/test_cuda_cpp_wrapper",
    "inductor/test_cuda_repro",
    "inductor/test_cudacodecache",
    "inductor/test_cudagraph_trees",
    "inductor/test_custom_lowering",
    "inductor/test_custom_post_grad_passes",
    "inductor/test_debug_trace",
    "inductor/test_dependencies",
    "inductor/test_efficient_conv_bn_eval",
    "inductor/test_extension_backend",
    "inductor/test_foreach",
    "inductor/test_fp8",
    "inductor/test_fused_attention",
    "inductor/test_fx_fusion",
    "inductor/test_group_batch_fusion",
    "inductor/test_indexing",
    "inductor/test_inductor_freezing",
    "inductor/test_inductor_utils",
    "inductor/test_inplacing_pass",
    "inductor/test_kernel_benchmark",
    "inductor/test_layout_optim",
    "inductor/test_max_autotune",
    "inductor/test_memory_planning",
    "inductor/test_minifier",
    "inductor/test_minifier_isolate",
    "inductor/test_mkldnn_pattern_matcher",
    "inductor/test_mmdecomp",
    "inductor/test_move_constructors_to_cuda",
    "inductor/test_pattern_matcher",
    "inductor/test_perf",
    "inductor/test_profiler",
    "inductor/test_select_algorithm",
    "inductor/test_smoke",
    "inductor/test_snode_runtime",
    "inductor/test_split_cat_fx_passes",
    "inductor/test_standalone_compile",
    "inductor/test_torchinductor",
    "inductor/test_torchinductor_codegen_dynamic_shapes",
    "inductor/test_torchinductor_dynamic_shapes",
    "inductor/test_torchinductor_opinfo",
    "inductor/test_triton_heuristics",
    "inductor/test_triton_wrapper",
    "inductor/test_unbacked_symints",
    "lazy/test_bindings",
    "lazy/test_debug_util",
    "lazy/test_extract_compiled_graph",
    "lazy/test_functionalization",
    "lazy/test_generator",
    "lazy/test_meta_kernel",
    "lazy/test_reuse_ir",
    "lazy/test_step_closures",
    "lazy/test_ts_opinfo",
    "nn/test_convolution",
    "nn/test_dropout",
    "nn/test_embedding",
    "nn/test_init",
    "nn/test_lazy_modules",
    "nn/test_module_hooks",
    "nn/test_multihead_attention",
    "nn/test_packed_sequence",
    "nn/test_parametrization",
    "nn/test_pooling",
    "nn/test_pruning",
    "optim/test_lrscheduler",
    "optim/test_optim",
    "optim/test_swa_utils",
    "profiler/test_memory_profiler",
    "profiler/test_profiler",
    "profiler/test_profiler_tree",
    "test_ao_sparsity",
    "test_autograd",
    "test_binary_ufuncs",
    "test_bundled_inputs",
    "test_comparison_utils",
    "test_compile_benchmark_util",
    "test_complex",
    "test_content_store",
    "test_cpp_api_parity",
    "test_cpp_extensions_aot_ninja",
    "test_cpp_extensions_aot_no_ninja",
    "test_cpp_extensions_jit",
    "test_cpp_extensions_open_device_registration",
    "test_cuda",
    "test_cuda_expandable_segments",
    "test_cuda_multigpu",
    "test_cuda_nvml_based_avail",
    "test_cuda_primary_ctx",
    "test_cuda_sanitizer",
    "test_cuda_trace",
    "test_custom_ops",
    "test_dataloader",
    "test_datapipe",
    "test_decomp",
    "test_deploy",
    "test_dispatch",
    "test_dlpack",
    "test_dynamic_shapes",
    "test_expanded_weights",
    "test_fake_tensor",
    "test_flop_counter",
    "test_foreach",
    "test_function_schema",
    "test_functional_autograd_benchmark",
    "test_functionalization",
    "test_functionalization_of_rng_ops",
    "test_futures",
    "test_fx",
    "test_fx_experimental",
    "test_fx_passes",
    "test_fx_reinplace_pass",
    "test_hub",
    "test_import_stats",
    "test_itt",
    "test_jit",
    "test_jit_autocast",
    "test_jit_disabled",
    "test_jit_fuser_te",
    "test_jit_llga_fuser",
    "test_jiterator",
    "test_legacy_vmap",
    "test_license",
    "test_logging",
    "test_masked",
    "test_maskedtensor",
    "test_matmul_cuda",
    "test_meta",
    "test_mkl_verbose",
    "test_mkldnn",
    "test_mkldnn_fusion",
    "test_mkldnn_verbose",
    "test_mobile_optimizer",
    "test_model_dump",
    "test_model_exports_to_core_aten",
    "test_modules",
    "test_monitor",
    "test_multiprocessing",
    "test_multiprocessing_spawn",
    "test_namedtensor",
    "test_namedtuple_return_api",
    "test_native_functions",
    "test_native_mha",
    "test_nestedtensor",
    "test_nn",
    "test_numba_integration",
    "test_numpy_interop",
    "test_openmp",
    "test_ops",
    "test_ops_fwd_gradients",
    "test_ops_gradients",
    "test_ops_jit",
    "test_optim",
    "test_out_dtype_op",
    "test_overrides",
    "test_package",
    "test_per_overload_api",
    "test_prims",
    "test_proxy_tensor",
    "test_pruning_op",
    "test_public_bindings",
    "test_python_dispatch",
    "test_pytree",
    "test_quantization",
    "test_reductions",
    "test_scatter_gather_ops",
    "test_schema_check",
    "test_segment_reductions",
    "test_serialization",
    "test_sparse",
    "test_sparse_csr",
    "test_sparse_semi_structured",
    "test_spectral_ops",
    "test_sympy_utils",
    "test_tensorexpr",
    "test_tensorexpr_pybind",
    "test_torch",
    "test_unary_ufuncs",
    "test_utils",
    "test_vulkan",
    "test_xnnpack_integration",
}

# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
#
# This lists exists so we can more easily add large numbers of failing tests,
dynamo_expected_failures = {
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attribute",
    "TestCppExtensionJIT.test_custom_compound_op_autograd",
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attributes",
    "TestCppExtensionOpenRgistration.test_open_device_registration",
    "TestAutogradFallback.test_supports_tensor_lists_mode_nothing",
    "TestAutogradFallback.test_post_autograd_returns_mix_of_requires_grad_tensors_mode_warn",
    "TestAutogradFallback.test_cpu_return_self_mode_warn",
    "TestAutogradFallback.test_base_does_not_require_grad_mode_warn",
    "TestAutogradFallback.test_undefined_grads_mode_nothing",
    "TestAutogradFallback.test_undefined_grads_mode_warn",
    "TestAutogradFallback.test_autograd_function_registered_to_cpu_mode_warn",
    "TestAutogradFallback.test_cpu_return_self_mode_nothing",
    "TestAutogradFallback.test_composite_registered_to_cpu_mode_nothing",
    "TestAutogradFallback.test_undefined_inputs_outputs_mode_nothing",
    "TestAutogradFallback.test_no_autograd_kernel_inplace_mode_nothing",
    "TestAutogradFallback.test_post_autograd_returns_leaf_mode_nothing",
    "TestAutogradFallback.test_inplace_on_tensor_that_does_not_require_grad_mode_nothing",
    "TestAutogradFallback.test_no_grad_mode_warn",
    "TestAutogradFallback.test_inplace_autograd_function_registered_to_cpu_mode_warn",
    "TestAutogradFallback.test_no_autograd_kernel_mode_warn",
    "TestAutogradFallback.test_base_does_not_require_grad_mode_nothing",
    "TestAutogradFallback.test_composite_registered_to_cpu_mode_warn",
    "TestAutogradFallback.test_post_autograd_returns_mix_of_requires_grad_tensors_mode_nothing",
    "TestAutogradFallback.test_no_autograd_kernel_inplace_mode_warn",
    "TestAutogradFallback.test_no_grad_mode_nothing",
    "TestAutogradFallback.test_no_autograd_kernel_mode_nothing",
    "TestAutogradFallback.test_supports_tensor_lists_mode_warn",
    "TestAutogradFallback.test_post_autograd_returns_leaf_mode_warn",
    "TestAutogradFallback.test_undefined_inputs_outputs_mode_warn",
    "TestAutogradFallback.test_inplace_on_tensor_that_does_not_require_grad_mode_warn",
    "TestAutogradFallback.test_inplace_autograd_function_registered_to_cpu_mode_nothing",
    "TestAutogradFallback.test_autograd_function_registered_to_cpu_mode_nothing",
    "TestFunctionalOptimParity.test_functional_optim_parity_sgd",
    "TestIndexingCPU.test_invalid_index_cpu",
    "NumpyTestsCPU.test_boolean_shape_mismatch_cpu",
    "TestIndexingCPU.test_empty_ndim_index_bool_cpu",
    "TestIndexingCPU.test_out_of_bound_index_cpu",
    "NumpyTestsCPU.test_index_no_floats_cpu",
    "TestIndexingCPU.test_zero_dim_index_cpu",
    "NumpyTestsCPU.test_empty_fancy_index_cpu",
    "TestIndexingCPU.test_index_cpu",
    "TestIndexingCPU.test_index_limits_cpu",
    "NumpyTestsCPU.test_boolean_indexing_weirdness_cpu",
    "TestLinalgCPU.test_inverse_cpu_float32",
    "TestLinalgCPU.test_matrix_rank_cpu_complex64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_float32",
    "TestLinalgCPU.test_inverse_cpu_complex128",
    "TestLinalgCPU.test_norm_dtype_cpu_complex128",
    "TestLinalgCPU.test_householder_product_cpu_float64",
    "TestLinalgCPU.test_linalg_lu_family_cpu_float32",
    "TestLinalgCPU.test_linalg_lu_family_cpu_float64",
    "TestLinalgCPU.test_addr_integral_cpu_int64",
    "TestLinalgCPU.test_norm_vector_cpu_float32",
    "TestLinalgCPU.test_solve_cpu_complex128",
    "TestLinalgCPU.test_lobpcg_torchscript_cpu_float64",
    "TestLinalgCPU.test_einsum_sublist_format_cpu_float64",
    "TestLinalgCPU.test_solve_cpu_float32",
    "TestLinalgCPU.test_addr_integral_cpu_int16",
    "TestLinalgCPU.test_norm_vector_cpu_float64",
    "TestLinalgCPU.test_einsum_random_cpu_complex128",
    "TestLinalgCPU.test_addmm_sizes_cpu_float64",
    "TestLinalgCPU.test_norm_dtype_cpu_float64",
    "TestLinalgCPU.test_addr_integral_cpu_int8",
    "TestLinalgCPU.test_einsum_random_cpu_float64",
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_complex64",
    "TestLinalgCPU.test_matrix_rank_cpu_float32",
    "TestLinalgCPU.test_pinv_cpu_float32",
    "TestLinalgCPU.test_addr_integral_cpu_uint8",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_complex128",
    "TestLinalgCPU.test_addr_integral_cpu_int32",
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_int64",
    "TestLinalgCPU.test_solve_cpu_complex64",
    "TestLinalgCPU.test_solve_cpu_float64",
    "TestLinalgCPU.test_addmm_sizes_cpu_float32",
    "TestLinalgCPU.test_norm_bfloat16_and_half_cpu_float16",
    "TestLinalgCPU.test_householder_product_cpu_complex64",
    "TestLinalgCPU.test_linalg_lu_family_cpu_complex128",
    "TestLinalgCPU.test_inverse_cpu_float64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_complex64",
    "TestLinalgCPU.test_pinv_cpu_complex64",
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_float32",
    "TestLinalgCPU.test_geqrf_cpu_complex128",
    "TestLinalgCPU.test_matrix_rank_cpu_complex128",
    "TestLinalgCPU.test_einsum_sublist_format_cpu_complex128",
    "TestLinalgCPU.test_geqrf_cpu_complex64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_float64",
    "TestLinalgCPU.test_linalg_lu_family_cpu_complex64",
    "TestLinalgCPU.test_matrix_rank_cpu_float64",
    "TestLinalgCPU.test_geqrf_cpu_float64",
    "TestLinalgCPU.test_householder_product_cpu_complex128",
    "TestLinalgCPU.test_geqrf_cpu_float32",
    "TestLinalgCPU.test_pinv_cpu_complex128",
    "TestLinalgCPU.test_pinv_cpu_float64",
    "TestLinalgCPU.test_householder_product_cpu_float32",
    "TestLinalgCPU.test_norm_bfloat16_and_half_cpu_bfloat16",
    "TestLinalgCPU.test_inverse_cpu_complex64",
    "TestModuleInitCPU.test_nn_FractionalMaxPool3d_cpu_float64",
    "TestModuleInitCPU.test_nn_PReLU_cpu_float64",
    "TestModuleInitCPU.test_nn_MultiLabelSoftMarginLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerEncoder_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyLinear_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_BCEWithLogitsLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_BatchNorm1d_cpu_float32",
    "TestModuleInitCPU.test_quantizable_LSTMCell_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose1d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyLinear_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv2d_cpu_float64",
    "TestModuleInitCPU.test_nn_PReLU_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_InstanceNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_ConvTranspose1d_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerEncoderLayer_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConvTranspose3d_cpu_float32",
    "TestModuleInitCPU.test_quantized_LeakyReLU_cpu_float32",
    "TestModuleInitCPU.test_quantized_GroupNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_RNNBase_cpu_float32",
    "TestModuleInitCPU.test_nn_FractionalMaxPool2d_cpu_float64",
    "TestModuleInitCPU.test_nn_LSTMCell_cpu_float64",
    "TestModuleInitCPU.test_nn_Embedding_cpu_float32",
    "TestModuleInitCPU.test_quantized_BatchNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_RNNCellBase_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose3d_cpu_float64",
    "TestModuleInitCPU.test_quantized_GroupNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_MultiLabelSoftMarginLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_GroupNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_RNNCell_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerEncoder_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm3d_cpu_float64",
    "TestModuleInitCPU.test_quantized_InstanceNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv3d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv2d_cpu_float32",
    "TestModuleInitCPU.test_nn_RNNCellBase_cpu_float32",
    "TestModuleInitCPU.test_quantized_Quantize_cpu_float32",
    "TestModuleInitCPU.test_nn_MultiheadAttention_cpu_float32",
    "TestModuleInitCPU.test_nn_TransformerEncoderLayer_cpu_float32",
    "TestModuleInitCPU.test_quantized_BatchNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm1d_cpu_float32",
    "TestModuleInitCPU.test_nn_RNNBase_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose2d_cpu_float64",
    "TestModuleInitCPU.test_nn_AdaptiveLogSoftmaxWithLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_Transformer_cpu_float64",
    "TestModuleInitCPU.test_quantizable_LSTM_cpu_float64",
    "TestModuleInitCPU.test_nn_BCEWithLogitsLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConv1d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_Embedding_cpu_float64",
    "TestModuleInitCPU.test_nn_FractionalMaxPool3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_GroupNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv3d_cpu_float64",
    "TestModuleInitCPU.test_nn_GRU_cpu_float32",
    "TestModuleInitCPU.test_qat_Conv3d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerDecoder_cpu_float64",
    "TestModuleInitCPU.test_nn_Conv3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm2d_cpu_float32",
    "TestModuleInitCPU.test_qat_Embedding_cpu_float32",
    "TestModuleInitCPU.test_nn_GRU_cpu_float64",
    "TestModuleInitCPU.test_quantized_LayerNorm_cpu_float32",
    "TestModuleInitCPU.test_quantizable_MultiheadAttention_cpu_float64",
    "TestModuleInitCPU.test_qat_Embedding_cpu_float64",
    "TestModuleInitCPU.test_nn_SyncBatchNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_Transformer_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_FractionalMaxPool2d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm2d_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv2d_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_Bilinear_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv2d_cpu_float64",
    "TestModuleInitCPU.test_qat_EmbeddingBag_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm1d_cpu_float32",
    "TestModuleInitCPU.test_quantizable_LSTMCell_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyBatchNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_NLLLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConv1d_cpu_float32",
    "TestModuleInitCPU.test_quantizable_MultiheadAttention_cpu_float32",
    "TestModuleInitCPU.test_nn_BCELoss_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerDecoderLayer_cpu_float32",
    "TestModuleInitCPU.test_nn_LayerNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_AdaptiveLogSoftmaxWithLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_CrossEntropyLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LayerNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_RNNCell_cpu_float32",
    "TestModuleInitCPU.test_nn_ConvTranspose1d_cpu_float64",
    "TestModuleInitCPU.test_nn_GRUCell_cpu_float64",
    "TestModuleInitCPU.test_nn_LSTMCell_cpu_float32",
    "TestModuleInitCPU.test_qat_Linear_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv2d_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm1d_cpu_float32",
    "TestModuleInitCPU.test_nn_TransformerDecoderLayer_cpu_float64",
    "TestModuleInitCPU.test_quantized_InstanceNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_SyncBatchNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_RNN_cpu_float32",
    "TestModuleInitCPU.test_nn_RNN_cpu_float64",
    "TestModuleInitCPU.test_quantizable_LSTM_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm3d_cpu_float32",
    "TestModuleInitCPU.test_quantized_Hardswish_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyBatchNorm1d_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm2d_cpu_float64",
    "TestModuleInitCPU.test_qat_EmbeddingBag_cpu_float64",
    "TestModuleInitCPU.test_quantized_BatchNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_CrossEntropyLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose3d_cpu_float64",
    "TestModuleInitCPU.test_quantized_Quantize_cpu_float64",
    "TestModuleInitCPU.test_nn_BCELoss_cpu_float32",
    "TestModuleInitCPU.test_nn_EmbeddingBag_cpu_float32",
    "TestModuleInitCPU.test_nn_LSTM_cpu_float64",
    "TestModuleInitCPU.test_nn_Linear_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_EmbeddingBag_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose2d_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_BatchNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_MultiMarginLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_MultiMarginLoss_cpu_float64",
    "TestModuleInitCPU.test_quantized_LayerNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_InstanceNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_Bilinear_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv1d_cpu_float64",
    "TestModuleInitCPU.test_nn_Conv1d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose2d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConvTranspose2d_cpu_float64",
    "TestModuleInitCPU.test_nn_MultiheadAttention_cpu_float64",
    "TestModuleInitCPU.test_nn_GRUCell_cpu_float32",
    "TestModuleInitCPU.test_quantized_LeakyReLU_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv2d_cpu_float64",
    "TestModuleInitCPU.test_nn_NLLLoss_cpu_float64",
    "TestModuleInitCPU.test_quantized_Hardswish_cpu_float32",
    "TestModuleInitCPU.test_nn_Linear_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose1d_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv1d_cpu_float32",
    "TestModuleInitCPU.test_nn_TransformerDecoder_cpu_float32",
    "TestModuleInitCPU.test_qat_Linear_cpu_float64",
    "TestModuleInitCPU.test_quantized_BatchNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LSTM_cpu_float32",
    "TestModuleInitCPU.test_qat_Conv1d_cpu_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc9_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_same_kind_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc1_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc16",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc1_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc5",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc3_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc6_out_dtype_float32",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_unsafe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc7",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_equiv_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc4",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc7_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc6_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_safe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc4_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc7_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc4_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc0",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc2_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc15",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc7_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc1_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc2_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc3_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc6_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc3",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc3_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc2_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc10",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc2_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc8_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_same_kind_ufunc0_out_dtype_float64",
    "TestUfuncDtypeKwd.test_binary_ufunc_dtype_and_out",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_same_kind_ufunc0_out_dtype_complex128",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_unsafe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc11",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc6_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc1",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc12",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc6_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc3_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc9",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc14",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc6",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc4_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc7_out_dtype_float32",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_unsafe_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc4_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc14_out_dtype_complex128",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_no_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc2_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_broadcast_ufunc0",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc3_out_dtype_complex128",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_safe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc7_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc1_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc6_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc8",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc1_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc4_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc13",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc2_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc2",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc1_out_dtype_float64",
    "TestIsScalar.test_is_not_scalar_value6",
    "TestGenericReductions.test_bad_axis_func0",
    "TestGenericReductions.test_bad_axis_func11",
    "TestGenericReductions.test_bad_axis_func7",
    "TestGenericReductions.test_bad_axis_func6",
    "TestGenericReductions.test_bad_axis_func2",
    "TestGenericCumSumProd.test_bad_axis_func1",
    "TestGenericReductions.test_bad_axis_func3",
    "TestGenericReductions.test_bad_axis_func4",
    "TestGenericReductions.test_bad_axis_func10",
    "TestGenericReductions.test_bad_axis_func5",
    "TestGenericReductions.test_bad_axis_func8",
    "TestGenericReductions.test_bad_axis_func1",
    "TestGenericCumSumProd.test_bad_axis_func0",
    "TestGenericReductions.test_bad_axis_func9",
    "TestShuffle.test_1d_use_numpy_True",
    "TestShuffle.test_1d_use_numpy_False",
    "TestShuffle.test_2d_use_numpy_True",
    "TestShuffle.test_2d_use_numpy_False",
    "TestArrayCreationCopyArgument.test_buffer_interface",
    "TestWritebackIfCopy.test_take_mode_raise",
    "TestArange.test_infinite",
    "TestArrayConstruction.test_array_empty",
    "TestAttributes.test_fill_readonly",
    "TestArrayAttributeDeletion.test_multiarray_writable_attributes_deletion",
    "TestMatmul.test_out_contiguous",
    "TestMinMax.test_scalar",
    "TestFromBuffer.test_basic_little_dtype2",
    "TestArrayCreationCopyArgument.test_striding_not_ok",
    "TestArange.test_require_range",
    "TestStats.test_dtype_from_input",
    "TestArange.test_nan_step",
    "TestWritebackIfCopy.test_argmin_with_out",
    "TestArrayAttributeDeletion.test_multiarray_not_writable_attributes_deletion",
    "TestLexsort.test_datetime",
    "TestMinMax.test_axis",
    "TestLexsort.test_mixed",
    "TestWritebackIfCopy.test_dot_out",
    "TestAttributes.test_fill_struct_array",
    "TestFromBuffer.test_empty",
    "TestAssignment.test_assignment_broadcasting",
    "TestMatmul.test_out_arg",
    "TestAttributes.test_set_stridesattr",
    "TestStats.test_out",
    "TestScalarIndexing.test_invalid_subscript",
    "TestWhere.test_error",
    "TestWritebackIfCopy.test_argmax_with_out",
    "TestBool.test_sum_2",
    "TestScalarIndexing.test_invalid_newaxis",
    "TestTake.test_out_overlap",
    "TestScalarIndexing.test_invalid_subscript_assignment",
    "TestFromBuffer.test_basic_little_dtype1",
    "TestWritebackIfCopy.test_choose_mod_raise",
    "TestAttributes.test_fill_max_uint64",
    "TestPutmask.test_byteorder_dtype_<i4",
    "TestPutmask.test_byteorder_dtype_>i4",
    "TestAttributes.test_stridesattr",
    "TestArange.test_zero_step",
    "TestStats.test_dtype_from_dtype",
    "TestArrayCreationCopyArgument.test_scalars",
    "TestConversion.test_to_int_scalar",
    "TestPutmask.test_record_array",
    "TestTake.test_raise",
    "TestFromBuffer.test_basic_little_dtype0",
    "TestMatmul.test_exceptions",
    "TestFlag.test_writeable_from_readonly",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_positional_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmin",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmin_np_method1",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmax",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_positional_arr_method_argmin_np_method1",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmax",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmin",
    "TestConvertDType.test_convert_np_dtypes_'int64'",
    "TestConvertDType.test_convert_np_dtypes_'uint8'",
    "TestConvertDType.test_convert_np_dtypes_bool",
    "TestConvertDType.test_convert_np_dtypes_'complex128'",
    "TestConvertDType.test_convert_np_dtypes_'float16'",
    "TestConvertDType.test_convert_np_dtypes_'int16'",
    "TestConvertDType.test_convert_np_dtypes_'int32'",
    "TestConvertDType.test_convert_np_dtypes_'int8'",
    "TestConvertDType.test_convert_np_dtypes_'float64'",
    "TestConvertDType.test_convert_np_dtypes_'float32'",
    "TestConvertDType.test_convert_np_dtypes_'complex64'",
    "TestConvertDType.test_convert_np_dtypes_'bool_'",
    "TestOneArr.test_asarray_list_func55",
    "TestOneArr.test_asarray_tensor_func65",
    "TestOneArr.test_asarray_tensor_func44",
    "TestOneArr.test_asarray_array_func59",
    "TestOneArr.test_asarray_array_func45",
    "TestOneArr.test_asarray_list_func70",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis_0",
    "TestSequenceOfArrays.test_single_array_func1",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis_1",
    "TestSequenceOfArraysToSingle.test_several_func6",
    "TestOneArr.test_asarray_list_func0",
    "TestOneArrAndAxis.test_andaxis_list_func9_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func8_axis_0",
    "TestOneArr.test_asarray_list_func36",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis_0",
    "TestOneArr.test_asarray_tensor_func15",
    "TestOneArr.test_asarray_array_func51",
    "TestOneArr.test_asarray_list_func16",
    "TestOneArrAndAxis.test_andaxis_tensor_func5_axis_0",
    "TestOneArrAndAxis.test_andaxis_tensor_func1_axis_1",
    "TestOneArr.test_asarray_tensor_func1",
    "TestOneArrAndAxesTuple.test_andtuple_list_func0_axes2",
    "TestOneArrAndAxis.test_andaxis_list_func6_axis_1",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis_-1",
    "TestSequenceOfArraysToSingle.test_several_func2",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis_1",
    "TestOneArr.test_asarray_array_func72",
    "TestOneArrAndShape.test_andshape_list_func0",
    "TestCtorNested.test_arrays_in_lists",
    "TestOneArr.test_asarray_tensor_func51",
    "TestOneArr.test_asarray_array_func0",
    "TestOneArr.test_asarray_array_func10",
    "TestOneArr.test_asarray_array_func43",
    "TestOneArrToScalar.test_toscalar_array_func2_np_func2",
    "TestOneArr.test_asarray_list_func3",
    "TestOneArr.test_asarray_array_func56",
    "TestArrayToSequence.test_asarray_array_func1",
    "TestOneArrAndShape.test_andshape_tensor_func4",
    "TestOneArr.test_asarray_list_func60",
    "TestDivmod.test_divmod_out",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis3",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis_0",
    "TestOneArrAndAxis.test_andaxis_list_func5_axis_0",
    "TestOneArr.test_asarray_tensor_func53",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis3",
    "TestOneArr.test_asarray_tensor_func73",
    "TestDivmod.test_divmod_no_out",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis_1",
    "TestOneArr.test_asarray_list_func58",
    "TestOneArrAndAxis.test_andaxis_tensor_func8_axis_0",
    "TestOneArr.test_asarray_array_func49",
    "TestOneArr.test_asarray_array_func60",
    "TestOneArr.test_asarray_tensor_func62",
    "TestOneArrAndAxesTuple.test_andtuple_tensor_func0_axes0",
    "TestOneArr.test_asarray_array_func22",
    "TestOneArr.test_asarray_list_func24",
    "TestOneArr.test_asarray_list_func15",
    "TestSequenceOfArrays.test_several_func2",
    "TestOneArr.test_asarray_tensor_func66",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis3",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis_0",
    "TestOneArrAndAxis.test_andaxis_list_func1_axis_-1",
    "TestOneArr.test_asarray_list_func32",
    "TestOneArr.test_asarray_list_func48",
    "TestOneArrToScalar.test_toscalar_array_func1_np_func1",
    "TestOneArr.test_asarray_list_func23",
    "TestOneArr.test_asarray_list_func65",
    "TestOneArr.test_asarray_tensor_func34",
    "TestOneArr.test_asarray_array_func57",
    "TestOneArr.test_asarray_list_func31",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis_0",
    "TestOneArr.test_asarray_array_func63",
    "TestOneArrAndAxis.test_andaxis_tensor_func9_axis_1",
    "TestOneArr.test_asarray_tensor_func0",
    "TestOneArr.test_asarray_list_func43",
    "TestOneArr.test_asarray_list_func62",
    "TestOneArrAndShape.test_andshape_array_func0",
    "TestSequenceOfArrays.test_several_func0",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis_-1",
    "TestOneArr.test_asarray_tensor_func29",
    "TestArrayToSequence.test_asarray_array_func0",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis3",
    "TestOneArr.test_asarray_array_func16",
    "TestOneArr.test_asarray_array_func68",
    "TestOneArr.test_asarray_list_func21",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis_1",
    "TestOneArr.test_asarray_array_func33",
    "TestOneArr.test_asarray_list_func13",
    "TestOneArr.test_asarray_list_func40",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis_0",
    "TestOneArrAndAxesTuple.test_andtuple_list_func0_axes0",
    "TestOneArr.test_asarray_list_func52",
    "TestOneArr.test_asarray_array_func42",
    "TestOneArr.test_asarray_list_func73",
    "TestOneArr.test_asarray_array_func24",
    "TestOneArr.test_asarray_list_func45",
    "TestOneArr.test_asarray_array_func38",
    "TestOneArr.test_asarray_array_func20",
    "TestOneArr.test_asarray_tensor_func45",
    "TestOneArr.test_asarray_array_func66",
    "TestOneArrAndAxis.test_andaxis_list_func2_axis_0",
    "TestOneArr.test_asarray_array_func11",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis3",
    "TestOneArrAndAxis.test_andaxis_list_func5_axis_-1",
    "TestOneArrAndShape.test_andshape_list_func1",
    "TestPythonArgsToArray.test_argstoarray_simple_func4_args4",
    "TestOneArr.test_asarray_tensor_func14",
    "TestOneArr.test_asarray_array_func48",
    "TestOneArr.test_asarray_list_func53",
    "TestOneArr.test_asarray_tensor_func24",
    "TestOneArr.test_asarray_list_func54",
    "TestOneArr.test_asarray_tensor_func33",
    "TestPythonArgsToArray.test_argstoarray_simple_func7_args7",
    "TestOneArrAndAxesTuple.test_andtuple_array_func0_axes1",
    "TestOneArrAndAxis.test_andaxis_list_func2_axis_1",
    "TestSequenceOfArrays.test_single_array_func0",
    "TestOneArr.test_asarray_tensor_func69",
    "TestSequenceOfArraysToSingle.test_several_func3",
    "TestOneArr.test_asarray_array_func36",
    "TestOneArr.test_asarray_list_func11",
    "TestCopyTo.test_copyto_typecast",
    "TestOneArrAndShape.test_andshape_tensor_func1",
    "TestOneArr.test_asarray_array_func71",
    "TestOneArrAndAxis.test_andaxis_list_func6_axis_0",
    "TestOneArrAndAxis.test_andaxis_tensor_func9_axis_0",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis_0",
    "TestOneArr.test_asarray_list_func72",
    "TestSequenceOfArraysToSingle.test_several_func4",
    "TestOneArrAndAxis.test_andaxis_tensor_func2_axis_0",
    "TestOneArrAndAxis.test_andaxis_list_func2_axis_-1",
    "TestOneArr.test_asarray_array_func34",
    "TestOneArr.test_asarray_array_func23",
    "TestOneArr.test_asarray_list_func20",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis_1",
    "TestOneArr.test_asarray_array_func41",
    "TestOneArr.test_asarray_list_func38",
    "TestOneArrAndAxis.test_andaxis_list_func5_axis_1",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis3",
    "TestOneArrToScalar.test_toscalar_array_func0_np_func0",
    "TestOneArr.test_asarray_tensor_func37",
    "TestOneArr.test_asarray_tensor_func20",
    "TestOneArr.test_asarray_tensor_func42",
    "TestOneArr.test_asarray_list_func67",
    "TestOneArr.test_asarray_list_func30",
    "TestOneArrAndAxis.test_andaxis_list_func4_axis_1",
    "TestSequenceOfArrays.test_several_func3",
    "TestOneArr.test_asarray_array_func54",
    "TestOneArrAndShape.test_andshape_list_func4",
    "TestOneArr.test_asarray_tensor_func2",
    "TestOneArr.test_asarray_tensor_func57",
    "TestOneArrAndAxis.test_andaxis_list_func9_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func4_axis_-1",
    "TestOneArr.test_asarray_array_func55",
    "TestOneArrAndAxis.test_andaxis_tensor_func3_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func5_axis_-1",
    "TestOneArr.test_asarray_list_func14",
    "TestOneArr.test_asarray_list_func29",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis_1",
    "TestOneArrAndShape.test_andshape_list_func3",
    "TestOneArr.test_asarray_tensor_func5",
    "TestOneArr.test_asarray_list_func68",
    "TestOneArr.test_asarray_tensor_func61",
    "TestSequenceOfArrays.test_single_list_func3",
    "TestOneArr.test_asarray_array_func21",
    "TestOneArr.test_asarray_list_func61",
    "TestOneArr.test_asarray_tensor_func55",
    "TestOneArr.test_asarray_tensor_func18",
    "TestOneArr.test_asarray_list_func50",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis_0",
    "TestOneArr.test_asarray_array_func62",
    "TestOneArr.test_asarray_tensor_func50",
    "TestOneArr.test_asarray_array_func6",
    "TestOneArr.test_asarray_list_func66",
    "TestOneArr.test_asarray_list_func59",
    "TestOneArr.test_asarray_tensor_func28",
    "TestShapeLikeToArray.test_shape_func3",
    "TestOneArr.test_asarray_array_func9",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis_0",
    "TestOneArrAndShape.test_andshape_array_func2",
    "TestPythonArgsToArray.test_argstoarray_simple_func2_args2",
    "TestOneArrAndShape.test_andshape_tensor_func0",
    "TestPythonArgsToArray.test_argstoarray_simple_func0_args0",
    "TestOneArr.test_asarray_array_func19",
    "TestOneArr.test_asarray_tensor_func39",
    "TestOneArr.test_asarray_array_func65",
    "TestSequenceOfArrays.test_single_list_func2",
    "TestOneArr.test_asarray_array_func31",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis_0",
    "TestOneArr.test_asarray_list_func2",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis3",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis_1",
    "TestDivmod.test_divmod_out_list",
    "TestOneArr.test_asarray_list_func19",
    "TestOneArrAndAxesTuple.test_andtuple_array_func0_axes2",
    "TestOneArr.test_asarray_array_func1",
    "TestOneArrAndAxis.test_andaxis_tensor_func4_axis_1",
    "TestOneArr.test_asarray_tensor_func43",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis_0",
    "TestOneArrAndAxesTuple.test_andtuple_tensor_func0_axes2",
    "TestOneArr.test_asarray_list_func10",
    "TestSequenceOfArrays.test_single_array_func3",
    "TestOneArr.test_asarray_tensor_func40",
    "TestSequenceOfArraysToSingle.test_several_func0",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis3",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis_-1",
    "TestOneArr.test_asarray_tensor_func35",
    "TestOneArr.test_asarray_tensor_func72",
    "TestOneArr.test_asarray_list_func18",
    "TestOneArr.test_asarray_tensor_func60",
    "TestOneArrAndAxis.test_andaxis_list_func3_axis_0",
    "TestOneArr.test_asarray_array_func37",
    "TestOneArr.test_asarray_array_func74",
    "TestNormalizations.test_unknown_args",
    "TestOneArr.test_asarray_array_func4",
    "TestOneArr.test_asarray_array_func58",
    "TestOneArrAndAxis.test_andaxis_list_func9_axis_0",
    "TestOneArr.test_asarray_tensor_func22",
    "TestOneArr.test_asarray_list_func56",
    "TestOneArrAndAxis.test_andaxis_list_func3_axis_1",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis_-1",
    "TestOneArr.test_asarray_tensor_func4",
    "TestPythonArgsToArray.test_argstoarray_simple_func6_args6",
    "TestOneArrAndAxis.test_andaxis_tensor_func9_axis_-1",
    "TestOneArr.test_asarray_tensor_func68",
    "TestOneArr.test_asarray_list_func27",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis_-1",
    "TestOneArr.test_asarray_array_func13",
    "TestOneArr.test_asarray_list_func6",
    "TestOneArr.test_asarray_array_func39",
    "TestOneArr.test_asarray_array_func73",
    "TestOneArr.test_asarray_tensor_func12",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis_-1",
    "TestOneArr.test_asarray_list_func17",
    "TestShapeLikeToArray.test_shape_func2",
    "TestOneArrAndAxis.test_andaxis_list_func4_axis_0",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis_1",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func8_axis_1",
    "TestOneArr.test_asarray_list_func33",
    "TestOneArrAndAxis.test_andaxis_tensor_func1_axis_-1",
    "TestOneArr.test_asarray_array_func18",
    "TestOneArr.test_asarray_tensor_func3",
    "TestOneArrAndShape.test_andshape_tensor_func2",
    "TestOneArr.test_asarray_list_func35",
    "TestOneArrAndAxis.test_andaxis_tensor_func3_axis_0",
    "TestOneArr.test_asarray_array_func70",
    "TestOneArrAndAxesTuple.test_andtuple_list_func0_axes1",
    "TestOneArrAndAxis.test_andaxis_list_func8_axis_-1",
    "TestOneArr.test_asarray_tensor_func59",
    "TestOneArr.test_asarray_array_func15",
    "TestOneArrAndAxis.test_andaxis_tensor_func6_axis_1",
    "TestOneArr.test_asarray_tensor_func38",
    "TestPythonArgsToArray.test_argstoarray_simple_func8_args8",
    "TestPythonArgsToArray.test_argstoarray_simple_func3_args3",
    "TestOneArr.test_asarray_array_func14",
    "TestPythonArgsToArray.test_argstoarray_simple_func5_args5",
    "TestOneArr.test_asarray_list_func26",
    "TestOneArr.test_asarray_list_func34",
    "TestOneArr.test_asarray_list_func4",
    "TestOneArr.test_asarray_tensor_func67",
    "TestOneArr.test_asarray_array_func3",
    "TestOneArr.test_asarray_array_func5",
    "TestOneArr.test_asarray_array_func52",
    "TestOneArr.test_asarray_tensor_func58",
    "TestOneArr.test_asarray_tensor_func48",
    "TestOneArr.test_asarray_array_func50",
    "TestOneArr.test_asarray_tensor_func47",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis3",
    "TestOneArrAndAxis.test_andaxis_tensor_func2_axis_1",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis3",
    "TestShapeLikeToArray.test_shape_func1",
    "TestOneArrAndAxis.test_andaxis_tensor_func4_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func8_axis_-1",
    "TestDefaultDtype.test_defaultdtype_defaults",
    "TestOneArr.test_asarray_list_func63",
    "TestOneArrAndShape.test_andshape_list_func2",
    "TestOneArr.test_asarray_array_func27",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis_0",
    "TestOneArr.test_asarray_list_func41",
    "TestSequenceOfArrays.test_single_tensor_func2",
    "TestOneArr.test_asarray_list_func39",
    "TestOneArr.test_asarray_tensor_func6",
    "TestOneArr.test_asarray_tensor_func25",
    "TestOneArr.test_asarray_array_func2",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis_1",
    "TestOneArr.test_asarray_tensor_func56",
    "TestOneArr.test_asarray_array_func69",
    "TestOneArr.test_asarray_list_func28",
    "TestOneArr.test_asarray_tensor_func26",
    "TestArrayToSequence.test_asarray_tensor_func1",
    "TestOneArr.test_asarray_array_func28",
    "TestPythonArgsToArray.test_argstoarray_simple_func1_args1",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis3",
    "TestOneArr.test_asarray_list_func44",
    "TestOneArr.test_asarray_array_func46",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis_1",
    "TestOneArr.test_asarray_tensor_func30",
    "TestOneArr.test_asarray_tensor_func16",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis3",
    "TestOneArr.test_asarray_tensor_func46",
    "TestOneArr.test_asarray_tensor_func10",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis_-1",
    "TestOneArr.test_asarray_list_func47",
    "TestSequenceOfArrays.test_single_tensor_func0",
    "TestOneArrAndAxesTuple.test_andtuple_array_func0_axes0",
    "TestOneArr.test_asarray_list_func12",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis3",
    "TestShapeLikeToArray.test_shape_func0",
    "TestOneArr.test_asarray_array_func61",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis_-1",
    "TestOneArrAndAxis.test_andaxis_list_func0_axis_0",
    "TestOneArr.test_asarray_tensor_func31",
    "TestOneArr.test_asarray_array_func67",
    "TestOneArr.test_asarray_list_func64",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis_1",
    "TestOneArr.test_asarray_array_func32",
    "TestOneArr.test_asarray_array_func8",
    "TestOneArr.test_asarray_list_func5",
    "TestOneArr.test_asarray_array_func17",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func5_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func0_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis_0",
    "TestOneArr.test_asarray_array_func64",
    "TestArrayToSequence.test_asarray_tensor_func0",
    "TestSequenceOfArrays.test_single_array_func2",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis_-1",
    "TestOneArr.test_asarray_list_func71",
    "TestOneArrAndAxesTuple.test_andtuple_tensor_func0_axes1",
    "TestOneArrAndAxis.test_andaxis_tensor_func1_axis_0",
    "TestOneArr.test_asarray_array_func44",
    "TestCopyTo.test_copyto_basic",
    "TestSequenceOfArrays.test_single_tensor_func1",
    "TestOneArr.test_asarray_tensor_func11",
    "TestSequenceOfArrays.test_several_func1",
    "TestOneArr.test_asarray_tensor_func74",
    "TestOneArr.test_asarray_tensor_func36",
    "TestOneArr.test_asarray_array_func53",
    "TestOneArr.test_asarray_tensor_func63",
    "TestOneArrAndShape.test_andshape_array_func3",
    "TestOneArr.test_asarray_list_func74",
    "TestOneArr.test_asarray_tensor_func49",
    "TestOneArrAndAxis.test_andaxis_tensor_func3_axis_1",
    "TestOneArr.test_asarray_tensor_func32",
    "TestOneArrAndAxis.test_andaxis_list_func1_axis_1",
    "TestOneArrAndAxis.test_andaxis_tensor_func4_axis_0",
    "TestOneArrAndShape.test_andshape_tensor_func3",
    "TestOneArr.test_asarray_tensor_func27",
    "TestOneArr.test_asarray_list_func22",
    "TestOneArr.test_asarray_list_func69",
    "TestOneArr.test_asarray_array_func26",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func6_axis_-1",
    "TestSequenceOfArrays.test_single_tensor_func3",
    "TestOneArrAndShape.test_andshape_array_func1",
    "TestOneArr.test_asarray_array_func25",
    "TestOneArrAndAxis.test_andaxis_tensor_func2_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis3",
    "TestOneArr.test_asarray_tensor_func41",
    "TestOneArrAndAxis.test_andaxis_tensor_func0_axis_1",
    "TestOneArr.test_asarray_list_func49",
    "TestOneArr.test_asarray_list_func57",
    "TestOneArrAndAxis.test_andaxis_tensor_func8_axis_1",
    "TestOneArr.test_asarray_tensor_func71",
    "TestSequenceOfArrays.test_single_list_func1",
    "TestPythonArgsToArray.test_argstoarray_simple_func9_args9",
    "TestOneArr.test_asarray_list_func37",
    "TestOneArrAndAxis.test_andaxis_tensor_func0_axis_0",
    "TestOneArr.test_asarray_array_func30",
    "TestOneArr.test_asarray_tensor_func21",
    "TestOneArr.test_asarray_array_func35",
    "TestOneArr.test_asarray_tensor_func64",
    "TestOneArr.test_asarray_list_func51",
    "TestOneArr.test_asarray_array_func47",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis_1",
    "TestOneArr.test_asarray_array_func29",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis_-1",
    "TestOneArr.test_asarray_tensor_func19",
    "TestOneArrAndAxis.test_andaxis_list_func1_axis_0",
    "TestOneArr.test_asarray_tensor_func17",
    "TestOneArrAndAxis.test_andaxis_list_func0_axis_1",
    "TestOneArr.test_asarray_tensor_func70",
    "TestOneArr.test_asarray_tensor_func54",
    "TestOneArr.test_asarray_tensor_func23",
    "TestOneArr.test_asarray_array_func7",
    "TestOneArr.test_asarray_array_func12",
    "TestOneArrAndAxis.test_andaxis_list_func3_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis_0",
    "TestOneArr.test_asarray_tensor_func13",
    "TestOneArrAndAxis.test_andaxis_tensor_func6_axis_0",
    "TestOneArrAndShape.test_andshape_array_func4",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis3",
    "TestOneArr.test_asarray_array_func40",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis_0",
    "TestOneArr.test_asarray_list_func42",
    "TestOneArrAndAxis.test_andaxis_tensor_func0_axis_-1",
    "TestOneArr.test_asarray_list_func25",
    "TestOneArr.test_asarray_tensor_func52",
    "TestOneArrAndAxis.test_andaxis_list_func6_axis_-1",
    "TestSequenceOfArraysToSingle.test_several_func1",
    "TestCopyTo.test_copytobcast",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis_-1",
    "TestSequenceOfArraysToSingle.test_several_func5",
    "TestOneArr.test_asarray_list_func1",
    "TestOneArr.test_asarray_list_func46",
    "TestSequenceOfArrays.test_single_list_func0",
    "TestCholesky.test_basic_property_shape3_dtype1",
    "TestCond.test_sq_cases",
    "TestNormInt64.test_bad_args",
    "TestQR.test_qr_empty_m_0_n_3",
    "TestMultiDot.test_dynamic_programming_optimization_and_out",
    "TestNormDouble.test_bad_args",
    "TestCholesky.test_basic_property_shape4_dtype1",
    "TestCholesky.test_basic_property_shape3_dtype2",
    "TestCholesky.test_basic_property_shape4_dtype0",
    "TestCholesky.test_basic_property_shape3_dtype0",
    "TestCond.test_empty_sq_cases",
    "TestCholesky.test_basic_property_shape1_dtype3",
    "TestQR.test_qr_empty_m_0_n_0",
    "TestQR.test_mode_raw",
    "TestMultiDot.test_two_arguments_and_out",
    "TestCholesky.test_basic_property_shape1_dtype2",
    "TestMultiDot.test_three_arguments_and_out",
    "TestNormDouble.test_axis",
    "TestCholesky.test_basic_property_shape1_dtype1",
    "TestCholesky.test_basic_property_shape2_dtype1",
    "TestMisc.test_generalized_raise_multiloop",
    "TestEigvalsh.test_invalid",
    "TestNormDouble.test_matrix_2x2",
    "TestCholesky.test_basic_property_shape0_dtype0",
    "TestMisc.test_byteorder_check",
    "TestCholesky.test_basic_property_shape4_dtype3",
    "TestCholesky.test_basic_property_shape2_dtype2",
    "TestCholesky.test_basic_property_shape3_dtype3",
    "TestNormInt64.test_axis",
    "TestCholesky.test_basic_property_shape2_dtype0",
    "TestCholesky.test_basic_property_shape0_dtype3",
    "TestQR.test_qr_empty_m_3_n_0",
    "TestEigh.test_invalid",
    "TestNormSingle.test_bad_args",
    "TestNormSingle.test_matrix_2x2",
    "TestNormSingle.test_axis",
    "TestCholesky.test_basic_property_shape1_dtype0",
    "TestCholesky.test_basic_property_shape4_dtype2",
    "TestMultiDot.test_too_few_input_arrays",
    "TestCholesky.test_basic_property_shape0_dtype2",
    "TestCholesky.test_basic_property_shape0_dtype1",
    "TestNormInt64.test_matrix_2x2",
    "TestCholesky.test_basic_property_shape2_dtype3",
    "TestFliplr.test_basic",
    "TestHistogram2d.test_binparameter_combination",
    "TestHistogram2d.test_all_outliers",
    "TestTriuIndicesFrom.test_exceptions",
    "TestTrilIndicesFrom.test_exceptions",
    "TestHistogram2d.test_asym",
    "TestDiag.test_failure",
    "TestVsplit.test_non_iterable",
    "TestVsplit.test_1D_array",
    "TestApplyAlongAxis.test_scalar_array",
    "TestDstack.test_non_iterable",
    "TestSplit.test_unequal_split",
    "TestPutAlongAxis.test_broadcast",
    "TestArraySplit.test_integer_0_split",
    "TestDsplit.test_2D_array",
    "TestTakeAlongAxis.test_invalid",
    "TestHsplit.test_0D_array",
    "TestDsplit.test_1D_array",
    "TestDsplit.test_non_iterable",
    "TestDsplit.test_0D_array",
    "TestHsplit.test_non_iterable",
    "TestColumnStack.test_non_iterable",
    "TestApplyAlongAxis.test_axis_insertion",
    "TestVsplit.test_0D_array",
    "TestExpandDims.test_repeated_axis",
    "TestExpandDims.test_axis_out_of_range",
    "TestApplyAlongAxis.test_0d_array",
    "TestHistogramdd.test_bins_errors",
    "TestHistogramdd.test_equal_edges",
    "TestHistogram.test_precision",
    "TestHistogramdd.test_finite_range",
    "TestHistogramdd.test_weights",
    "TestHistogram.test_error_binnum_type",
    "TestHistogram.test_finite_range",
    "TestHistogramdd.test_inf_edges",
    "TestHistogramdd.test_bins_error_2",
    "TestHistogramdd.test_simple",
    "TestHistogram.test_one_bin",
    "TestHistogram.test_unsigned_monotonicity_check",
    "TestQuantile.test_quantile_monotonic_method_weibull",
    "TestGradient.test_badargs",
    "TestRot90.test_basic",
    "TestDiff.test_axis",
    "TestQuantile.test_quantile_monotonic_method_median_unbiased",
    "TestGradient.test_values",
    "TestCov.test_aweights",
    "TestQuantile.test_quantile_monotonic_method_interpolated_inverted_cdf",
    "TestQuantile.test_quantile_monotonic_method_inverted_cdf",
    "TestPercentile.test_keepdims_out_q1_axis_1",
    "TestSortComplex.test_sort_real_type_in_g_type_out_G",
    "TestMedian.test_keepdims_out_axis2",
    "TestMeshgrid.test_invalid_arguments",
    "TestGradient.test_specific_axes",
    "TestPercentile.test_keepdims_out_q_7_axis4",
    "TestPercentile.test_keepdims_out_q1_axis4",
    "TestDelete.test_slices",
    "TestPercentile.test_extended_axis_invalid",
    "TestGradient.test_second_order_accurate",
    "TestMedian.test_keepdims_out_axis0",
    "TestDiff.test_prepend",
    "TestMedian.test_keepdims_out_axis_1",
    "TestPercentile.test_keepdims_out_q1_axis0",
    "TestQuantile.test_quantile_monotonic_method_averaged_inverted_cdf",
    "TestMedian.test_keepdims_out_axis4",
    "TestBincount.test_with_incorrect_minlength",
    "TestSortComplex.test_sort_real_type_in_H_type_out_F",
    "TestDiff.test_n",
    "TestMeshgrid.test_indexing",
    "TestQuantile.test_quantile_monotonic_method_closest_observation",
    "TestFlip.test_axes",
    "TestPercentile.test_keepdims_out_q1_axis3",
    "TestPercentile.test_keepdims_out_q_7_axis0",
    "TestMedian.test_keepdims_out_axis3",
    "TestCov.test_fweights",
    "TestDiff.test_append",
    "TestPercentile.test_scalar_q",
    "TestMedian.test_extended_axis_invalid",
    "TestMedian.test_out",
    "TestPercentile.test_keepdims_out_q_7_axis2",
    "TestPercentile.test_keepdims_out_q1_axis2",
    "TestQuantile.test_quantile_monotonic_method_hazen",
    "TestPercentile.test_keepdims_out_q_7_axis3",
    "TestPercentile.test_keepdims_out_q_7_axis_1",
    "TestPercentile.test_api",
    "TestQuantile.test_quantile_monotonic_method_normal_unbiased",
    "TestSetOps.test_in1d_mixed_dtype_dtype11_dtype21_kind_table",
    "TestSetOps.test_in1d_mixed_dtype_dtype10_dtype20_kind0",
    "TestSetOps.test_in1d_mixed_dtype_dtype10_dtype20_kind_table",
    "TestSetOps.test_ediff1d_forbidden_type_casts_ary1_prepend1_append1_expected_to_begin",
    "TestSetOps.test_in1d_mixed_dtype_dtype11_dtype21_kind0",
    "TestSetOps.test_in1d_mixed_dtype_dtype11_dtype21_kind_sort",
    "TestSetOps.test_in1d_table_timedelta_fails",
    "TestUnique.test_unique_axis_errors",
    "TestSetOps.test_setdiff1d",
    "TestSetOps.test_in1d_mixed_dtype_dtype10_dtype20_kind_sort",
    "TestSetOps.test_in1d_timedelta_kind_sort",
    "TestSetOps.test_in1d_timedelta_kind0",
    "TestUnique.test_unique_axis",
    "TestConstant.test_check_constant_float3",
    "TestConstant.test_check_constant_pad_2d",
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_c8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_no",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_exceptions",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_c8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_equiv",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_large_concatenate_axis_None",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_concatenate",  # torch_np/numpy_tests/core/test_shape_base
    "TestVstack.test_empty_input",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_i8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_no",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_no",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f4_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestVstack.test_non_iterable",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_i8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f4_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f4_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_c8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_c8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f4_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f4_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_c8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_bad_out_shape",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_equiv",  # torch_np/numpy_tests/core/test_shape_base
    "TestHstack.test_non_iterable",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_c8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestHstack.test_empty_input",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f4_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_equiv",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestNegative.test_exceptions",  # torch_np/numpy_tests/core/test_scalarmath
    "TestPower.test_modular_power",  # torch_np/numpy_tests/core/test_scalarmath
    "TestBaseMath.test_lower_align",  # torch_np/numpy_tests/core/test_scalarmath
    "TestArrayFromScalar.test_integers_np_longlong_t26",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_intc_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_t15_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_longlong_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_byte_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_short_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_int__np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestScalarTypeNames.test_names_reflect_attributes_t4",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t1",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t7",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t5",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t9",  # torch_np/numpy_tests/core/test_numerictypes
    "TestIsSubDType.test_both_abstract",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t6",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t2",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t8",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t0",  # torch_np/numpy_tests/core/test_numerictypes
    "TestIsSubDType.test_nondtype_nonscalartype",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t3",  # torch_np/numpy_tests/core/test_numerictypes
    "TestClip.test_clip_inplace_array",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_require_each",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_simple_int32",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_inplace_01",  # torch_np/numpy_tests/core/test_numeric
    "TestStdVar.test_out_scalar",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_int32_inout_casting_unsafe",  # torch_np/numpy_tests/core/test_numeric
    "TestMoveaxis.test_errors",  # torch_np/numpy_tests/core/test_numeric
    "TestNonzeroAndCountNonzero.test_count_nonzero_axis",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_memory_overlap",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_func_takes_out",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_noncontig_inplace",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_type_cast_12",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_int64_out",  # torch_np/numpy_tests/core/test_numeric
    "TestRollaxis.test_exceptions",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_inplace_02",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_C_and_F_simul",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-1",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_complex",  # torch_np/numpy_tests/core/test_numeric
    "TestBoolArray.test_logical_not_abs",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_out",  # torch_np/numpy_tests/core/test_numeric
    "TestBroadcast.test_broadcast_single_arg",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_unknown_requirement",  # torch_np/numpy_tests/core/test_numeric
    "TestBoolArray.test_logical_and_or_xor",  # torch_np/numpy_tests/core/test_numeric
    "TestBroadcast.test_broadcast_error_kwargs",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-9",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-10",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_type_cast_10",  # torch_np/numpy_tests/core/test_numeric
    "TestOuterMisc.test_outer_out_param",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_inplace_simple",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_transposed",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_simple",  # torch_np/numpy_tests/core/test_numeric
    "TestCross.test_broadcasting_shapes",  # torch_np/numpy_tests/core/test_numeric
    "TestIndexing.test_index_no_floats",  # torch_np/numpy_tests/core/test_indexing
    "TestBooleanIndexing.test_boolean_indexing_weirdness",  # torch_np/numpy_tests/core/test_indexing
    "TestBooleanIndexing.test_bool_as_int_argument_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestBroadcastedAssignments.test_simple_broadcasting_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestFloatNonIntegerArgument.test_non_integer_argument_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestIndexing.test_slicing_no_floats",  # torch_np/numpy_tests/core/test_indexing
    "TestBroadcastedAssignments.test_prepend_not_one",  # torch_np/numpy_tests/core/test_indexing
    "TestFloatNonIntegerArgument.test_reduce_axis_float_index",  # torch_np/numpy_tests/core/test_indexing
    "TestEinsum.test_different_paths_dtype_f",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_D",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_e",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_fixed_collapsingbug",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_combined_views_mapping",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_B",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_cfloat64",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_broadcast",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_int32",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_b",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_fixedstridebug",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_out_is_res",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_subscript_range",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_float64",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_float32",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_cfloat128",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_small_boolean_arrays",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_i",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_d",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_l",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_h",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_misc",  # torch_np/numpy_tests/core/test_einsum
    "TestMisc.test_overlap",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_int64",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_failed_on_p9_and_s390x",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_F",  # torch_np/numpy_tests/core/test_einsum
    "TestDLPack.test_dtype_passthrough_dtype4",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_23",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_12",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_27",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_32",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_from_dlpack_refcount",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype2",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_2",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_ndim0",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_1",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_17",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_13",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_14",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype7",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype9",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_29",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dunder_dlpack_refcount",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_15",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_non_contiguous",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype3",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_30",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_6",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_7",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype6",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype5",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_4",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_31",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_from_torch",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_24",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_21",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype8",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_28",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_3",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_10",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_0",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_16",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_18",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_20",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_11",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_25",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_5",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_22",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dlpack_device",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_9",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype0",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype1",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_19",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_26",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_8",  # torch_np/numpy_tests/core/test_dlpack
    "WeakTest.test_make_weak_keyed_dict_from_weak_keyed_dict",  # test_weak
    "WeakKeyDictionaryTestCase.test_update",  # test_weak
    "TestViewOpsLAZY.test_advanced_indexing_assignment_lazy",  # test_view_ops
    "TestOldViewOpsCPU.test_crow_col_indices_cpu",  # test_view_ops
    "TestViewOpsLAZY.test_advanced_indexing_nonview_lazy",  # test_view_ops
    "TestTypePromotionCPU.test_alpha_mismatch_cpu",  # test_type_promotion
    "TestTypePromotionCPU.test_alternate_result_cpu",  # test_type_promotion
    "TestTypeHints.test_doc_examples",  # test_type_hints
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float32_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_different_datatypes_kernel2_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float64_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float16_cpu_float16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape1_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_different_datatypes_kernel1_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestAttnMasksCPU.test_is_causal_and_mask_fails_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_attention_math_with_negative_scale_kernel0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape2_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestTransformersCPU.test_train_with_is_causal_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_bfloat16_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPAFailureModesCPU.test_invalid_inputs_1_dimensional_inputs_kernel0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPAFailureModesCPU.test_invalid_inputs_different_datatypes_kernel0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape3_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float32_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_1_dimensional_inputs_kernel2_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_1_dimensional_inputs_kernel1_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float64_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float16_cpu_float16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_bfloat16_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestAssertCloseSparseCOO.test_matching_coalesced",  # test_testing
    "TestImports.test_circular_dependencies",  # test_testing
    "TestAssertCloseSparseCSR.test_mismatching_crow_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_mismatching_row_indices_msg",  # test_testing
    "TestAssertCloseSparseCOO.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseQuantized.test_matching_per_channel",  # test_testing
    "TestTestParametrizationDeviceTypeCPU.test_ops_decorator_applies_op_and_param_specific_decorators_cpu",  # test_testing
    "TestAssertCloseSparseCOO.test_matching_uncoalesced",  # test_testing
    "TestAssertCloseSparseCSR.test_matching",  # test_testing
    "TestAssertCloseSparseBSR.test_mismatching_crow_indices_msg",  # test_testing
    "TestAssertCloseSparseBSR.test_matching",  # test_testing
    "TestAssertCloseQuantized.test_mismatching_is_quantized",  # test_testing
    "TestAssertCloseSparseCOO.test_mismatching_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_mismatching_ccol_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseCSC.test_mismatching_row_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_matching",  # test_testing
    "TestAssertCloseSparseCSC.test_matching",  # test_testing
    "TestAssertCloseSparseCSR.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseBSR.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseCSC.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseBSR.test_mismatching_col_indices_msg",  # test_testing
    "TestAssertCloseSparseCOO.test_mismatching_nnz",  # test_testing
    "TestAssertCloseSparseCSR.test_mismatching_col_indices_msg",  # test_testing
    "TestAssertCloseQuantized.test_mismatching_qscheme",  # test_testing
    "TestAssertCloseQuantized.test_matching_per_tensor",  # test_testing
    "TestAssertCloseSparseCSC.test_mismatching_ccol_indices_msg",  # test_testing
    "TestTensorBoardUtils.test_to_HWC",  # test_tensorboard
    "TestTensorBoardEmbedding.test_embedding",  # test_tensorboard
    "TestTensorProtoSummary.test_float_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_image_without_channel",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_smoke",  # test_tensorboard
    "TestTensorBoardUtils.test_numpy_vid_uint8",  # test_tensorboard
    "TestTensorProtoSummary.test_complex_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_image_with_one_channel",  # test_tensorboard
    "TestTensorBoardEmbedding.test_embedding_64",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_domain_discrete",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_wrong_parameter",  # test_tensorboard
    "TestTensorBoardSummary.test_video",  # test_tensorboard
    "TestTensorProtoSummary.test_int_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_number",  # test_tensorboard
    "TestTensorBoardWriter.test_writer",  # test_tensorboard
    "TestTensorProtoSummary.test_empty_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_string",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_bool",  # test_tensorboard
    "TestTensorBoardSummary.test_uint8_image",  # test_tensorboard
    "TestBufferProtocolCPU.test_shared_buffer_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_complex128",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_complex128",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_int32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_complex128",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_complex128",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_float32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_int16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_int16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_float32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_int8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_int16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_int8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_int8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_float16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_complex64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_complex128",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_complex64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_int32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_int16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_complex128",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_int16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_int16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_int8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_int8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_int8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_float32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_bool",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_tensor_factory_type_inference_cpu",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_byte_to_int_cpu",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_block_diag_cpu",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_int16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_complex128",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_complex128",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_complex128",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_bool",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_and_offset_cpu_int32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_int64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_uint8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_offset_cpu_int8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_bfloat16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_float64",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_constructor_dtypes_cpu",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_int64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_float64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_complex128",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_same_type_cpu_int8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_int16",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_tensor_factory_copy_var_cpu",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_int64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_from_buffer_cpu_bool",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_requires_grad_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_int16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_float64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_with_count_cpu_float16",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_invalid_positional_args_cpu_complex64",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_complex64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_alias_from_buffer_cpu_int8",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_float32",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_shared_buffer_cpu_float16",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_cartesian_prod_cpu",  # test_tensor_creation_ops
    "TestSubclass.test_parametrization_non_wrapper_tensor_leave_parametrized_True",  # test_subclass
    "TestSubclass.test_module_optimization_non_wrapper_tensor",  # test_subclass
    "TestSubclass.test_serialization_non_wrapper_tensor_as_param_True",  # test_subclass
    "TestSubclass.test_module_optimization_sparse_tensor",  # test_subclass
    "TestSubclass.test_param_invariants_non_wrapper_tensor_tensor_requires_grad_False",  # test_subclass
    "TestSubclass.test_param_invariants_sparse_tensor_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_param_invariants_diag_tensor_below_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_param_invariants_diag_tensor_below_tensor_requires_grad_False",  # test_subclass
    "TestSubclass.test_param_invariants_non_wrapper_tensor_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_parametrization_non_wrapper_tensor_leave_parametrized_False",  # test_subclass
    "TestSubclass.test_type_propagation_non_wrapper_tensor_as_param_False",  # test_subclass
    "TestSubclass.test_module_optimization_diag_tensor_below",  # test_subclass
    "TestSubclass.test_parametrization_base_tensor_leave_parametrized_True",  # test_subclass
    "TestSubclass.test_type_propagation_non_wrapper_tensor_as_param_True",  # test_subclass
    "TestSubclass.test_parametrization_base_tensor_leave_parametrized_False",  # test_subclass
    "TestSubclass.test_param_invariants_sparse_tensor_tensor_requires_grad_False",  # test_subclass
    "TestStatelessFunctionalAPI.test_reparametrize_module_fail_reset_to_original_torch_func",  # test_stateless
    "TestStatelessFunctionalAPI.test_reparametrized_module_change_parametrization_original_stateless",  # test_stateless
    "TestStatelessFunctionalAPI.test_reparametrized_module_change_parametrization_original_torch_func",  # test_stateless
    "TestStatelessFunctionalAPI.test_reparametrize_module_fail_reset_to_original_stateless",  # test_stateless
    "TestSortAndSelectCPU.test_isin_cpu_int32",  # test_sort_and_select
    "TestSortAndSelectCPU.test_sort_overflow_cpu_int16",  # test_sort_and_select
    "TestSortAndSelectCPU.test_topk_quantized_scalar_input_cpu",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_float64",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_uint8",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_int8",  # test_sort_and_select
    "TestSortAndSelectCPU.test_topk_arguments_cpu",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_int16",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_int64",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_float32",  # test_sort_and_select
    "TestShapeOpsCPU.test_flip_cpu_float64",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_float32",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_complex64",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_float16",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_complex128",  # test_shape_ops
    "TestShapeOpsCPU.test_clamp_cpu_int64",  # test_shape_ops
    "TestShapeOpsCPU.test_clamp_propagates_nans_cpu",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_bfloat16",  # test_shape_ops
    "TestShapeOpsCPU.test_clamp_cpu_float32",  # test_shape_ops
}

dynamo_skips = {
    "TestMatmulOperator.test_matmul_raises",
    "TestMatmulOperator.test_exceptions",
    "TestMatmulOperator.test_matmul_inplace",
    "TestMethods.test_diagonal",
    "TestMethods.test_searchsorted_complex",
    "TestMethods.test_round",
    "TestMethods.test_searchsorted_type_specific_2",
    "TestMethods.test_dot",
    "TestMethods.test_dot_out_mem_overlap",
    "TestMethods.test_partition_iterative",
    "TestMethods.test_trace",
    "TestMethods.test_matmul_out",
    "TestMethods.test_transpose",
    "TestMethods.test_conjugate",
    "TestMethods.test_choose_2",
    "TestMethods.test_size_zero_memleak",
    "TestMethods.test_searchsorted_with_invalid_sorter",
    "TestMethods.test_choose",
    "TestMethods.test_conjugate_out",
    "TestMethods.test_compress",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmin_np_method1",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmin",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmax",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmax",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmin",
    "TestIsreal.test_fail",  # known py311 fail
    "TestIscomplexobj.test_basic",  # known py311 fail
    "TestIsrealobj.test_basic",  # known py311 fail
    "TestIsreal.test_pass",  # known py311 fail
    "TestIscomplex.test_pass",  # known py311 fail
    "TestIscomplexobj.test_list",  # known py311 fail
    "TestDiag.test_matrix",  # known py311 fail
    "TestVander.test_dtypes",  # known py311 fail
    "TestDstack.test_generator",  # known py311 fail
    "TestColumnStack.test_generator",  # known py311 fail
    "TestCov.test_complex",  # known py311 fail
    "TestSortComplex.test_sort_complex",  # known py311 fail
    "TestCorrCoef.test_xy",  # known py311 fail
    "TestCov.test_xy",  # known py311 fail
    "TestCorrCoef.test_complex",  # known py311 fail
    "TestUnique.test_simple_complex",  # known py311 fail
    "TestDigitize.test_casting_error",  # known py311 fail
    "TestConstant.test_check_constant",  # known py311 fail
    "TestFFTShift.test_fft_n",  # known py311 fail
    "TestHstack.test_generator",  # known py311 fail
    "TestVstack.test_generator",  # known py311 fail
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_I_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_I_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_L_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_Q_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_Q_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_P_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_P_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_H_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_H_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_L_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestCorrelate.test_complex",  # known py311 fail
    "TestStdVarComplex.test_basic",  # known py311 fail
    "TestEinsum.test_broadcasting_dot_cases",  # known py311 fail
    "WeakTest.test_make_weak_keyed_dict_from_dict",  # known py311 fail
    "TestViewOpsCPU.test_as_strided_gradients_cpu",  # known py311 fail
    "TestViewOpsLAZY.test_as_strided_gradients_lazy",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape3_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape0_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape1_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape2_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape3_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape0_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape2_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape1_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape0_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape2_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape0_cpu",  # known py311 fail
    "TestTransformersCPU.test_decoder_padding_and_src_mask_bool_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape3_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape3_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape2_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape1_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape1_cpu",  # known py311 fail
    "TestFrameworkUtils.test_filtering_env_var",  # known py38 fail
    "TestAsArrayCPU.test_default_device_cpu",  # known py38 fail
    "TestAsArrayCPU.test_astensor_consistency_cpu",  # known py311 fail
    "TestTensorCreationCPU.test_vander_types_cpu_complex128",  # known py311 fail
    "TestTensorCreationCPU.test_vander_types_cpu_complex64",  # known py311 fail
    "TestTensorCreationCPU.test_torch_polar_cpu_float32",  # known py311 fail
    "TestTensorCreationCPU.test_torch_polar_cpu_float64",  # known py311 fail
}
