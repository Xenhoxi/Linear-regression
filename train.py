import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def main() -> None:
    data = read_dataset()
    if not data.empty:
        Theta0, Theta1 = linear_regression(data)
        create_csv(Theta0, Theta1)
        calcul_precision(data, Theta0, Theta1)


def read_dataset() -> pd.DataFrame:
    try:
        data = pd.read_csv("data.csv")
        # print("Dataset successfully open !")
        return (data)
    except FileNotFoundError:
        print("Unexpected error, impossible to read the dataset !")
        return (None)


def display_graph(mileage, price, Theta0, Theta1) -> None:
    plt.figure(figsize=(12, 8))
    plt.scatter(mileage, price, color='blue')
    plt.axline((0, Theta0), slope=Theta1, color='red')
    plt.ylim(0, 10000)
    plt.title("Linear regression model: Car price based on mileage", fontweight='bold')
    plt.xlabel("Mileage (km)", fontweight='bold')
    plt.ylabel("Price (€)", fontweight='bold')
    equation = f"(Theta0 = {Theta0:.2f}, Theta1 = {Theta1:.2f})"
    plt.text(0, 9500, equation, fontsize=12, fontweight='bold',color='green')
    plt.savefig("Linear regression plot")
    plt.show()
    plt.clf()


def linear_regression(data):
    learn_rate = .1
    Theta0 = 0
    Theta1 = 0
    real_price = np.array(data["price"])
    real_mileage = np.array(data["km"])

    # Standardized price and mileage
    price = standadized(real_price)
    mileage = standadized(real_mileage)

    for i in range(0, 100):
        # Values are standardized
        predicted_price = Theta0 + (Theta1 * mileage)
        cost = np.sum((predicted_price - price) ** 2)
        # print(f"Result of cost: {cost}")

        derive_Theta0 = (1 / price.size) * np.sum(predicted_price - price)
        derive_Theta1 = (1 / price.size) * np.sum((predicted_price - price) * mileage)
        # print(f"derive theta0: {derive_Theta0:.4f}, derive theta1: {derive_Theta1:.4f}")

        Theta0 = Theta0 - learn_rate * derive_Theta0
        Theta1 = Theta1 - learn_rate * derive_Theta1

    # Values are no more standardized back to og values
    Theta0, Theta1 = unstandardized(Theta0, Theta1, real_price, real_mileage)
    display_graph(real_mileage, real_price, Theta0, Theta1)
    return (Theta0, Theta1)


def unstandardized(Theta0, Theta1, real_price, real_mileage):
    normal_Theta1 = (Theta1 * real_price.std()) / real_mileage.std()
    normal_Theta0 = Theta0 * real_price.std() + real_price.mean() - normal_Theta1 * real_mileage.mean()
    return normal_Theta0, normal_Theta1


def calcul_precision(data: pd.DataFrame, Theta0: float, Theta1: float):
    print(f"Theta0: {Theta0} et Theta1: {Theta1}")
    price = np.array(data["price"])
    mileage = np.array(data["km"])
    predicted_price = Theta0 + (Theta1 * mileage)
    rss = np.sum((price - predicted_price) ** 2)
    tss = np.sum((price - price.mean()) ** 2)
    print(f"Precision: {round((1 - (rss / tss)) * 100, 3)}%")
    price = np.array(data["price"])
    predicted_price = Theta0 + (Theta1 * mileage)
    mae = 1 / price.size * np.sum((abs(predicted_price - price)))
    print("Result of MAE cost:", round(mae))
    rmse = 1 / price.size * np.sum((predicted_price - price) ** 2)
    print("Result of RMSE cost:", round(math.sqrt(rmse)))


def standadized(values):
    return (values - values.mean()) / values.std()


def create_csv(Theta0, Theta1):
    with open("training_result.csv", "w") as file:
        file.write("Theta0,Theta1\n")
        file.write(f"{Theta0},{Theta1}")


if __name__ == "__main__":
    main()
