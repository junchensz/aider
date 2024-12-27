from aider.coders import Coder
from aider.models import Model
from dotenv import load_dotenv
from aider.io import InputOutput

load_dotenv(dotenv_path="test.env")
fnames = ["mall-admin/src/main/java/com/macro/mall/service/impl/OmsOrderServiceImpl.java"]

# model = Model("openai/hf:deepseek-ai/DeepSeek-V2.5")
model = Model("deepseek")

io = InputOutput(yes=True,llm_history_file='.aider.llm.history')
# Create a coder object
coder = Coder.create(main_model=model,io=io, fnames=fnames,suggest_shell_commands=False, unit_mode=True)

# This will execute one instruction on those files and then return
coder.run("make unit test for mall-admin/src/main/java/com/macro/mall/service/impl/UmsAdminServiceImpl.java, need include success, fail and edge cases")

coder.run("/test mvn -f mall-admin/pom.xml test ")

coder.run("fix that, we should not modify any files in mall-admin/src/main")