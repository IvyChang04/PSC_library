import subprocess


def run_command(command):
    """Utility function to run a shell command."""
    try:
        print("Executing command:", " ".join(command))
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing:", " ".join(command))
        print(e)


# Experiment for table 3
datasets_table3 = ["Firewall", "Letter"]
for dataset in datasets_table3:
    run_command(
        [
            "python",
            "table_3/main.py",
            "--methods",
            "sc",
            "psc",
            "kmeans",
            "--dataset",
            dataset,
            "--size",
            "-1",
        ]
    )

# Experiment for table 4
data_sizes_table4 = ["15000", "30000", "45000", "60000"]
for data_size_table4 in data_sizes_table4:
    run_command(
        [
            "python",
            "Firewall_table4/main.py",
            "--methods",
            "sc",
            "--size",
            data_size_table4,
        ]
    )

run_command(
    [
        "python",
        "Firewall_table4/main.py",
        "--methods",
        "psc",
        "--size",
        "60000",
    ]
)

# Experiment for table 5
data_sizes_table5 = ["50000", "200000", "1040000"]
for data_size_table5 in data_sizes_table5:
    run_command(
        [
            "python",
            "NIDS_table5/main.py",
            "--methods",
            "psc",
            "--size",
            data_size_table5,
            "--ratio",
            "0.9",
        ]
    )
    run_command(
        ["python", "NIDS_table5/main.py", "--methods", "sc", "--size", data_size_table5]
    )

# Experiment for Figure 1
datasets_figure1 = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
]
for dataset in datasets_figure1:
    run_command(["python", "Synthesis_dataset/figure1.py", "--dataset", dataset])

# Experiment for Figure 2
run_command(["python", "Synthesis_dataset/figure2.py"])
