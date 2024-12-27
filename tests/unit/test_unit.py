from aider.coders import Coder
from aider.models import Model
from dotenv import load_dotenv


if __name__ == "__main__":
    # This is a list of files to add to the chat
    load_dotenv(dotenv_path="test.env")
    fnames = ["greeting.py"]

    # model = Model("openai/hf:deepseek-ai/DeepSeek-V2.5")
    model = Model("openai/deepseek-chat")

    # Create a coder object
    coder = Coder.create(main_model=model, fnames=fnames,suggest_shell_commands=False, unit_mode=True)

    # This will execute one instruction on those files and then return
    coder.run("make a script that prints hello world")

    # Send another instruction
    coder.run("make it say goodbye")

    # You can run in-chat "/" commands too
    coder.run("/tokens")
