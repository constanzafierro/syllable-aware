from src.CheckModel import CheckModel


def main():
    path_model = "../models/lstm_model_180301.0133.h5"
    path_experiment = "../models/experiment/experimentT500Tw30Ts470_setting_token.txt"
    path_tokenselector = "../models/experiment/experimentT500Tw30Ts470_setting_tokenSelector.txt"

    path_test = "./data/test2.txt"

    checkmodel = CheckModel(path_model, path_experiment, path_tokenselector)
    seed = "declaro"
    text = checkmodel.predict_text(nwords=20)

    print("="*50)
    print("Seed to generate text '{}'".format(seed))
    print("="*50)
    print("text generate {}".format(text))
    print("="*50)
    ppl = checkmodel.perplexity(path_test)
    print("perplexity per word in file {} is {}".format(path_test, ppl))

if __name__ == '__main__':
    main()