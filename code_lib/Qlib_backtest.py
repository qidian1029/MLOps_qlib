from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


def qlib_backtest(config,dataset,model,rid):
    config["port_analysis_config"]["strategy"]["kwargs"]["model"]= model
    config["port_analysis_config"]["strategy"]["kwargs"]["dataset"]= dataset
    port_analysis_config = config["port_analysis_config"]
    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        # ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder,port_analysis_config, "day")
        par.generate()

    pred_df = recorder.load_object("pred.pkl")
    # pred_df_dates = pred_df.index.get_level_values(level='datetime')
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    #positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")


    from code_lib import ML_data
    result_path = config["folders"]["result"]
    #ML_data.save_df_to_csv(pred_df,result_path)
    #ML_data.save_df_to_csv(report_normal_df,result_path)
    #ML_data.save_df_to_csv(analysis_df,result_path)

    return pred_df,report_normal_df,analysis_df