import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def create_apuf(n_stages, n_instances):
    return np.random.randn(n_instances, n_stages)

def generate_crps(apuf, n_challenges, noise_std):
    n_instances, n_stages = apuf.shape
    challenges = np.random.choice([1, -1], size=(n_challenges, n_stages))
    delays = challenges @ apuf.T
    responses = np.sign(np.sum(delays, axis=1))
    responses = np.round(responses).astype(int)
    responses = apply_noise(responses, noise_std)
    return challenges, responses

def apply_noise(responses, noise_std):
    noise = np.random.normal(0, noise_std, size=responses.shape)
    noisy_responses = responses + noise
    return np.sign(noisy_responses).astype(int)

def attack_apuf(apuf, n_challenges, n_attack_challenges, noise_std):
    challenges, responses = generate_crps(apuf, n_challenges, noise_std)
    lr = LogisticRegression(max_iter=10000).fit(challenges, responses)
    svm = SVC(kernel="linear", max_iter=10000).fit(challenges, responses)
    attack_challenges, attack_responses = generate_crps(apuf, n_attack_challenges, noise_std)
    lr_pred_responses = lr.predict(attack_challenges)
    svm_pred_responses = svm.predict(attack_challenges)
    lr_accuracy = accuracy_score(attack_responses, lr_pred_responses)
    svm_accuracy = accuracy_score(attack_responses, svm_pred_responses)
    return lr_accuracy, svm_accuracy

def main():
    n_experiments = 10
    n_challenges = 10000
    n_attack_challenges = 1000
    noise_stds = [0, 0.25, 0.5, 0.75, 1]

    stage_range = [64]  # Only use 64-stage APUF
    instance_range = [100]

    lr_results = {n: [] for n in noise_stds}
    svm_results = {n: [] for n in noise_stds}

    for noise_std in noise_stds:
        print(f"Noise standard deviation: {noise_std}")

        for n_stages in stage_range:
            for n_instances in instance_range:
                print(f"  Number of stages: {n_stages}, number of instances: {n_instances}")

                total_lr_accuracy = 0
                total_svm_accuracy = 0

                for _ in range(n_experiments):
                    apuf = create_apuf(n_stages, n_instances)
                    lr_accuracy, svm_accuracy = attack_apuf(apuf, n_challenges, n_attack_challenges, noise_std)
                    total_lr_accuracy += lr_accuracy
                    total_svm_accuracy += svm_accuracy

                avg_lr_accuracy = total_lr_accuracy / n_experiments
                avg_svm_accuracy = total_svm_accuracy / n_experiments

                print("    Average Logistic Regression accuracy: {:.2f}%".format(avg_lr_accuracy * 100))
                print("    Average Support Vector Machine accuracy: {:.2f}%".format(avg_svm_accuracy * 100))

                lr_results[noise_std].append(avg_lr_accuracy)
                svm_results[noise_std].append(avg_svm_accuracy)

                print("-----------------------------------------------------------")

    # Plot the average prediction rates for LR and SVM for each noise level
    fig, axs = plt.subplots(len(noise_stds), figsize=(10, 20), sharex=True, sharey=True)
    fig.suptitle('Average prediction rate for 64-stage APUF models with varying Gaussian noise standard deviations')

    for idx, noise_std in enumerate(noise_stds):
        axs[idx].plot(stage_range, lr_results[noise_std], label='Logistic Regression', marker='o')
        axs[idx].plot(stage_range, svm_results[noise_std], label='Support Vector Machine', marker='o')
        axs[idx].set_title(f"Noise standard deviation: {noise_std}")
        axs[idx].set_xlabel('Number of stages')
        axs[idx].set_ylabel('Prediction rate')
        axs[idx].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('prediction_rate_with_gaussian_noise_std_64_stage.png')
    plt.show()

if __name__ == "__main__":
    main()

