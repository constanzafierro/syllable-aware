from src.evaluationModel import evaluationModel


def main():
    path_model = "../models/lstm_model_180301.0133.h5"
    path_experiment = "../models/experiment/experimentT500Tw30Ts470_setting_token.txt"
    path_tokenselector = "../models/experiment/experimentT500Tw30Ts470_setting_tokenSelector.txt"

    evaluationmodel_ = evaluationModel(path_model, path_experiment, path_tokenselector)
    evaluationmodel_.predict_text(nwords=20)


if __name__ == '__main__':
    main()