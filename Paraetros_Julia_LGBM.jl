# Definir las variables importantes
param_local = Dict(
    "future" => [202108],
    "final_train" => Dict(
        "undersampling" => 1.0,
        "clase_minoritaria" => ["BAJA+1", "BAJA+2"],
        "training" => [202106, 202105, 202104, 202103, 202102, 202101],
        # [202106, 202105, 202104, 202103, 202102, 202101,
        #    202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012]
    ),
    "train" => Dict(
        "training" => [202106, 202105, 202104, 202103, 202102, 202101],
        #[202104, 202103, 202102, 202101,
        #202012, 202011, 202005, 202006, 202007, 202008, 202009, 202010],
        "validation" => [202105],
        "testing" => [202106],
        "undersampling" => 0.75,
        "clase_minoritaria" => ["BAJA+1", "BAJA+2"]
    ),
    "lgb_param" => Dict(
        "boosting" => "gbdt",
        "objective" => "binary",
        "metric" => "custom",
        "first_metric_only" => true,
        "boost_from_average" => true,
        "feature_pre_filter" => false,
        "force_row_wise" => true,
        "verbosity" => -1, #catedra pone -100 pero cambio para ocultar msj Warning
        "max_depth" => -1,
        "min_gain_to_split" => 0.0,
        "min_sum_hessian_in_leaf" => 0.001,
        "lambda_l1" => 0.0,
        "lambda_l2" => 0.0,
        "max_bin" => 31,
        "num_iterations" => 9999,
        "bagging_fraction" => 1.0,
        "pos_bagging_fraction" => 1.0,
        "neg_bagging_fraction" => 1.0,
        "is_unbalance" => false,
        "scale_pos_weight" => 1.0,
        "drop_rate" => 0.1,
        "max_drop" => 50,
        "skip_drop" => 0.5,
        "extra_trees" => false,
        "learning_rate" => 0.02,#, 0.3],
        "feature_fraction" => 0.5,# 0.9],
        "num_leaves" => 200, #8,
        "min_data_in_leaf" => 1000#, 10000]
    )
)

