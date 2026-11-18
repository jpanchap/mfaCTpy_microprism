# Microprism pipeline setup

## 1. Get the tools

Install Git, Miniconda (or use an existing Conda installation), and an editor such as VS Code.

Open Terminal on macOS/Linux or a Conda-enabled prompt on Windows. Check that both commands print a version number:

```sh
git --version
conda --version
```

## 2. Download the code

Run these commands one at a time:

```sh
git clone https://github.com/jpanchap/mfaCTpy_microprism.git
cd mfaCTpy_microprism
```

Keep this terminal in the downloaded folder, which contains `environment.yml` and `src/`.

## 3. Install and check dependencies

Run each command after the previous one finishes:

```sh
conda env create -f environment.yml
conda activate mfaCT
python --version
python -m pip check
python -m tkinter
```

Installation may take several minutes, including at **Installing pip dependencies**.

You should see Python **3.11.x**, **No broken requirements found**, and a small window titled **tk**. Click **QUIT** to close it.

In future terminal sessions, run `conda activate mfaCT` to use this environment.

## 4. Set up your editor if not already

In VS Code:
1. Open the downloaded `mfaCTpy_microprism` folder and install the **Python** extension from Microsoft.
2. Open the Command Palette with **Cmd+Shift+P** (Mac) or **Ctrl+Shift+P** (Windows/Linux).
3. Choose **Python: Select Interpreter**, then select **mfaCT**.

For another IDE, select `mfaCT` in its Python interpreter settings. To find the exact interpreter path, run this in your activated terminal:

```sh
python -c "import sys; print(sys.executable)"
```

In the editor's terminal, also run `conda activate mfaCT`.

## Common setup errors

| Problem | What to do |
| --- | --- |
| Git or Conda command is unrecognized | Confirm it is installed and reopen your terminal. On Windows, use a Conda-enabled prompt. |
| `environment.yml` cannot be found | Run `cd mfaCTpy_microprism` from the folder where you cloned the code. |
| `EnvironmentNameNotFound` | Run `conda env list` and activate the exact listed name. Check that installation finished successfully. |
| `mfaCT` already exists | Run `conda activate mfaCT`, then the three Python checks in step 3. |
| Installation fails or the Tk window does not open | Send the command and complete error message, along with your operating system. |
