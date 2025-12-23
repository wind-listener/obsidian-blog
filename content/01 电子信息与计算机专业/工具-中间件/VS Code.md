```json

'num'
{

// Use IntelliSense to learn about possible attributes.

// Hover to view descriptions of existing attributes.

// For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387

"version": "0.2.0",

"configurations": [

{

"name": "Python Debugger: Current File",

"type": "debugpy",

"request": "launch",

"program": "${file}",

"console": "integratedTerminal",

"args": [

"--model", "glm-4-private-FC",

"--test-category", "rest"

]

},

{

"name": "Python Debugger:executable_parallel_multiple_function",

"type": "debugpy",

"request": "launch",

"program": "${file}",

"console": "integratedTerminal",

"args": [

"--model", "ipo:glm-4-air-biz",

"--test-category", "executable_parallel_multiple_function"

]

}

]

}
```