# Chatbot basé sur Streamlit et OpenAI

Ce projet est une interface Streamlit pour un chatbot utilisant le modèle GPT-3-5 d'OpenAI et une suite d'outils fournis par `langchain`.

Son objectif est de répondre aux questions concernant la matière du cours. Le dossier `theorie` contient un exemple des fichiers sur lequel le chatbot repondra. Suffit de remplacer ces fichiers text par les votres pour que le chatbot y répond!

## Prérequis

- Python 3.x
- Sur Windows, installer: [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/)
    - Assurez-vous de choisir "Desktop development with C++" lors de l'installation.

## Installation

1. Clonez le dépôt.

    ```bash
    git clone https://github.com/AiWaldoh/students-chatbot.git
    ```

2. Naviguez vers le répertoire du projet:

    ```bash
    cd student-chatbot
    ```

3. Installez `venv` (si ce n'est pas déjà fait) et créez un nouvel environnement virtuel :

    ```bash
    python -m venv monenv
    ```

    **Note :** Vous pouvez remplacer `monenv` par le nom que vous souhaitez donner à votre environnement virtuel.

4. Activez l'environnement virtuel :

    - **Windows :**

    ```bash
    .\monenv\Scripts\activate
    ```

    - **macOS et Linux :**

    ```bash
    source monenv/bin/activate
    ```

5. Installez les dépendances à partir du fichier `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

6. Configurez votre clé API OpenAI en créant un fichier `.env` et en y ajoutant:

    ```bash
    OPENAI_API_KEY=votre_clé_api
    ```

## Exécution

Lancez l'application Streamlit en utilisant:

```bash
streamlit run main.py```
