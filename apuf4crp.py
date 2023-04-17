import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def create_apuf(n_stages, n_instances):
    return np.random.randn(n_instances, n_stages)

def generate_crps(apuf, n_challenges, noise):
    n_instances, n_stages = apuf.shape
    challenges = np.random.choice([1, -1], size=(n_challenges, n_stages))
    delays = challenges @ apuf.T
    responses = np.sign(np.sum(delays, axis=1))
    responses = np.round(responses).astype(int)
    responses = apply_noise(responses, noise)
    return challenges, responses

def apply_noise(responses, noise_level):
    n_responses = len(responses)
    flip_indices = np.random.choice([True, False], size=n_responses, p=[noise_level, 1 - noise_level])
    responses[flip_indices] = -responses[flip_indices]
    return responses

def attack_apuf(apuf, n_challenges, n_attack_challenges, noise):
    challenges, responses = generate_crps(apuf, n_challenges, noise)
    lr = LogisticRegression(max_iter=10000).fit(challenges, responses)
    svm = SVC(kernel="linear", max_iter=10000).fit(challenges, responses)
    attack_challenges, attack_responses = generate_crps(apuf, n_attack_challenges, noise)
    lr_pred_responses = lr.predict(attack_challenges)
    svm_pred_responses = svm.predict(attack_challenges)
    lr_accuracy = accuracy_score(attack_responses, lr_pred_responses)
    svm_accuracy = accuracy_score(attack_responses, svm_pred_responses)
    return lr_accuracy, svm_accuracy

def main():
    n_experiments = 10
    noise = 0
    training_samples = [100, 500, 4000, 8000]

    stage_range = [64]  # Only use 64-stage APUF
    instance_range = [100]

    lr_results = []
    svm_results = []

    for n_challenges in training_samples:
        print(f"Number of training samples: {n_challenges}")

        for n_stages in stage_range:
            for n_instances in instance_range:
                print(f"  Number of stages: {n_stages}, number of instances: {n_instances}")

                total_lr_accuracy = 0
                total_svm_accuracy = 0

                for _ in range(n_experiments):
                    apuf = create_apuf(n_stages, n_instances)
                    lr_accuracy, svm_accuracy = attack_apuf(apuf, n_challenges, n_challenges, noise)
                    total_lr_accuracy += lr_accuracy
                    total_svm_accuracy += svm_accuracy

                avg_lr_accuracy = total_lr_accuracy / n_experiments
                avg_svm_accuracy = total_svm_accuracy / n_experiments

                print("    Average Logistic Regression accuracy: {:.2f}%".format(avg_lr_accuracy * 100))
                print("    Average Support Vector Machine accuracy: {:.2f}%".format(avg_svm_accuracy * 100))

                lr_results.append(avg_lr_accuracy)
                svm_results.append(avg_svm_accuracy)

                print("-----------------------------------------------------------")

        # Plot the average prediction rates for LR and SVM for each number of training samples
    plt.plot(training_samples, lr_results, label='Logistic Regression', marker='o')
    plt.plot(training_samples, svm_results, label='Support Vector Machine', marker='o')

    plt.title('Prediction rates for 64-stage APUF with varying training sample sets')
    plt.xlabel('Number of training samples')
    plt.ylabel('Prediction rate')
    plt.legend()

    plt.savefig('prediction_rate_with_training_samples_64_stage.png')
    plt.show()

if __name__ == "__main__":
    main()
