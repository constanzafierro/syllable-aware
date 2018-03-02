from src.evaluationModel import evaluationModel


def main():
    path_model = "./models/lstm_model_180301.0133.h5"
    path_experiment = "./data/experimentT500Tw30Ts470.txt"

    evaluationModel_ = evaluationModel(path_model, path_experiment)
    evaluationModel_.predict_text(nwords=20)


if __name__ == '__main__':
    main()