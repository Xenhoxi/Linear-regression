import pandas as pd


def main():
    mileage = None

    try:
        while not mileage:
            mileage = input_mileage()
        mileage = int(mileage)
        result = predict_price(mileage)
        print(f"The estimate price is {result:.2f}$")
    except (KeyboardInterrupt, EOFError):
        pass


def read_dataset(file) -> pd.DataFrame:
    try:
        data = pd.read_csv(file, dtype=float)
        return (data)
    except FileNotFoundError:
        print("No training result found !")
        return (pd.DataFrame())


def input_mileage():
    try:
        mileage = input("What's the mileage you want to estimate the price ? ")
        if int(mileage) < 0:
            raise (ValueError)
        return (mileage)
    except ValueError:
        print("Input a valid mileage (Km).")
        return (None)


def predict_price(mileage):
    Theta0 = 0
    Theta1 = 0
    training = read_dataset("training_result.csv")
    if not training.empty:
        if "Theta0" in training.columns and "Theta1" in training.columns:
            Theta0 = training["Theta0"].iloc[0]
            Theta1 = training["Theta1"].iloc[0]
    return (Theta0 + (Theta1 * mileage))


if __name__ == "__main__":
    main()
