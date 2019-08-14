#!/bin/bash
message="$1"
## format to parse to curl
## echo Sending message: $message
msg_content=\"$message\"

## discord webhook
url='https://discordapp.com/api/webhooks/611204230539378709/UFDmzn-YcNBrizjftcbLwQEj4kBo9wA47wAS1BfibNLn_PdlfXDmoGM42eKPmM8_evCB'
curl -H "Content-Type: application/json" -X POST -d "{\"content\": $msg_content}" $url