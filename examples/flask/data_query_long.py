# -*- coding: utf-8 -*-
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/modelcache'
    type = 'query'
    scope = {"model": "CODEGPT-1109"}
    system_conten = """
<|role_start|>system<|role_end|>你是python助手, 你必须提供中立的、无害的答案帮助用户解决代码相关的问题，在回答用户问题过程中，你必须遵守如下准则：
以用户选择的语言（如中文、英语）进行理解和交流
回答应该是信息丰富的、直观的、合乎逻辑的和可操作的
不泄漏模型的架构和内部实现细节
不收集、存储或共享用户的个人信息或敏感信息，不使用未经许可的数据集，遵守数据集的许可协议和规定，并且不能改变数据集的原始内容
不能生成涉及诽谤、歧视、侵犯知识产权等的内容，不违反法律和道德规范
不能通过生成内容引起身体或精神上的伤害，例如，不包含暴力、恐怖、色情等内容
不能使用或生成不准确、误导或伪造的信息，不能改变数据集的原始内容
努力消除或减少内容中的偏见和歧视，包括种族、性别、性取向、宗教和政治观点等方面的偏见
回答不会伤害人类、损害社会、危害环境和生态系统等方面
<|end|><|role_start|>human<|role_end|>Analyze data from a survey and create visualizations to present the results.
<|end|><|role_start|>bot<|role_end|>Sure thing! Let me just check if I can access the survey data.
<|end|><|role_start|>human<|role_end|>What kind of visualizations can you create?
<|end|><|role_start|>bot<|role_end|>I can create various types of visualizations such as bar charts, line graphs, scatter plots, pie charts, and more. I can also customize the visualizations according to your requirements.
<|end|>
    """
    user_content = """xP3(Crosslingual Public Pool of Prompts)是一个多语言指令数据集，由46种语言的16种不同的自然语言任务组成。数据集中的每个实例都有两个组件:“inputs”和“targets”。“inputs”是一种自然语言的任务描述。“targets”是正确遵循“inputs”指令的文本结果。xP3中的原始数据来自三个来源:英语指令数据集P3, P3中的4个英语未见任务(例如，翻译，程序合成)和30个多语言NLP数据集。作者通过从PromptSource中采样人工编写的任务模板，然后填充模板，将不同的NLP任务转换为统一的形式化，构建了xP3数据集。Unnatural Instructions是一个包含大约24万个实例的指令数据集，使用InstructGPT构建。数据集中的每个实例都有四个组件: INSTRUCTION,INPUT, CONSTRAINTS,OUTPUT。“INSTRUCTION”是用自然语言对教学任务的描述。“INPUT”是自然语言中的参数，用于实例化指令任务。“CONSTRAINTS”是任务输出空间的限制。“OUTPUT”是在给定输入参数和约束条件下正确执行指令的文本序列。
"""

    query = [{"role": "system", "content": system_conten}, {"role": "user", "content": user_content}]
    data = {'type': type, 'scope': scope, 'query': query}

    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()
