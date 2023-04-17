import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def create_apuf(n_stages, n_instances):
    return np.random.randn(n_instances, n_stages)

def generate_crps(apuf, n_challenges):
    n_instances, n_stages = apuf.shape
    challenges = np.random.choice([1, -1], size=(n_challenges, n_stages))
    delays = challenges @ apuf.T
    responses = np.sign(np.sum(delays, axis=1))
    responses = np.round(responses).astype(int)
    return challenges, responses

def attack_apuf(apuf, n_challenges, n_attack_challenges):
    challenges, responses = generate_crps(apuf, n_challenges)
    lr = LogisticRegression(max_iter=10000).fit(challenges, responses)
    svm = SVC(kernel="linear", max_iter=10000).fit(challenges, responses)
    attack_challenges, attack_responses = generate_crps(apuf, n_attack_challenges)
    lr_pred_responses = lr.predict(attack_challenges)
    svm_pred_responses = svm.predict(attack_challenges)
    lr_accuracy = accuracy_score(attack_responses, lr_pred_responses)
    svm_accuracy = accuracy_score(attack_responses, svm_pred_responses)
    return lr_accuracy, svm_accuracy

def main():
    n_experiments = 10
    n_challenges = 10000
    n_attack_challenges = 1000

    stage_range = [64, 128, 256]
    instance_range = [100]

    lr_results = {}
    svm_results = {}

    for n_stages in stage_range:
        for n_instances in instance_range:
            print(f"Number of stages: {n_stages}, number of instances: {n_instances}")

            total_lr_accuracy = 0
            total_svm_accuracy = 0

            for _ in range(n_experiments):
                apuf = create_apuf(n_stages, n_instances)
                lr_accuracy, svm_accuracy = attack_apuf(apuf, n_challenges, n_attack_challenges)
                total_lr_accuracy += lr_accuracy
                total_svm_accuracy += svm_accuracy

            avg_lr_accuracy = total_lr_accuracy / n_experiments
            avg_svm_accuracy = total_svm_accuracy / n_experiments

            print("  Average Logistic Regression accuracy: {:.2f}%".format(avg_lr_accuracy * 100))
            print("  Average Support Vector Machine accuracy: {:.2f}%".format(avg_svm_accuracy * 100))

            lr_results.setdefault('accuracy', []).append(avg_lr_accuracy)
            svm_results.setdefault('accuracy', []).append(avg_svm_accuracy)

            print("-----------------------------------------------------------")

    # Plot the average prediction rates for LR and SVM
    plt.plot(stage_range, lr_results['accuracy'], label='Logistic Regression')
    plt.plot(stage_range, svm_results['accuracy'], label='Support Vector Machine')
    plt.xlabel('Number of stages')
    plt.ylabel('Prediction rate')
    plt.title('Average prediction rate for different APUF models')
    plt.legend()
    plt.savefig('prediction_rate.png')
    plt.show()

if __name__ == "__main__":
    main()
