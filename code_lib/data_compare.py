from code_lib import ML_plt

def comepare(config):
    compare_config = {
        "report_normal_df" : True,
        "analysis_df" : False
    }
    result_path = config["folders"]["result"]
    if config["compare"] is not None:
        compare_config.update(config["compare"])

    if compare_config["report_normal_df"] :
        ML_plt.compare_report_normal_df(result_path + "/report_normal_df",result_path)
    if compare_config["analysis_df"]:
        ML_plt.compare_analysis_df(result_path+"/analysis_df",result_path)

    pass