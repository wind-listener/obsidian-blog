**[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL), the first comprehensive evaluation on the LLM's ability to call functions and tools**.
Quick Links:
- Live Leaderboard: [Website](https://gorilla.cs.berkeley.edu/leaderboard.html)
- BFCL Evaluation Dataset: [HuggingFace Dataset 🤗](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- Gradio Demo: [HuggingFace Space 🤗](https://huggingface.co/spaces/gorilla-llm/berkeley-function-calling-leaderboard)
- Reproducibility: [Github Code](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- OpenFunctions-v2 (6.91B) on HuggingFace 🤗: [gorilla-llm/gorilla-openfunctions-v2](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2)
---
# 简介
数据集特点
 - 2k question-function-answer pairs
 -  multiple languages (python, java, javascript, restAPI), 
 - diverse application domains and complex use cases 
	 - multiple function calls where the LLM needs to select one or more functions from multiple functions provided
	 - parallel function calls that the LLM needs to make multiple function calls together
- **BFCL includes 100 Java, 50 JavaScript, 70 REST API, 100 SQL, and 1,680 Python on various simple, parallel, multiple, executable functions calling scenarios as well as function relevance detection**

# 能力评估的九个纬度 nine distinct categories: 
1. function relevance detection
2. AST (Abstract Syntax Tree) tree analysis：simple, parallel, multiple, parallel multiple
3. execution function call verification ： simple, parallel, multiple, parallel multiple
![[Pasted image 20240605120708.png]]

# 数据集构成
![[Pasted image 20240605115855.png]]
- **Python**: Simple Function, Multiple Function, Parallel Function, Parallel Multiple Function
	- Simple Function,    给一个，调用一个
	- Multiple Function,   给多个API document， 选择一个调用
	- Parallel Function,  给一个，但是需要并行调用多次
	- Parallel Multiple Function 给多个，选择调用哪些和各自调用几次
- **Non-Python**: Chatting Capability, Function Relevance Detection, REST API, SQL, Java, Javascript
	- Chatting Capability,
	- Function Relevance Detection, 
	- REST API,  编写AST太复杂，可以直接检查http code
	- SQL,  没有作为Leaderboard的评价内容，因为语法灵活，多种写法调用可以获得相同结果
	- Java+Javascript 只有AST，执行太复杂

![[Pasted image 20240605144420.png]]