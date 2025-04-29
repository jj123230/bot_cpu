<h1> CPU chat bot </h1>
Download the whole bot_cpu folder as a bot, bot_cpu.py will be the main file to activate the bot

1. Install the env by conda env create -f folder_path\bot_cpu\requirement.yml
2. Set your host_ip, port_ip, env and DB_info in setting_config.py
3. Set your db structure in config
4. Edit the folder path in test.txt, use folder_path/bot_cpu/bot_cpu.py folder_path to activate the CPU chat bot

The API will respond in CPU environment, because not everyone afford GPU, so I create a chat bot which is light enough to fit in CPU environment

<h2> Main functions </h2>
http://host_ip/port_ip/updatefaq: refresh the threshold in the model <br>
http://host_ip/port_ip/updatemodel: train model <br>
http://host_ip/port_ip/load_model: load model in local folder <br>
<br>
http://host_ip/port_ip/dscbot: chat bot that answers faq, parameter: text= original text, lan= language <br>
