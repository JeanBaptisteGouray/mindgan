#!/bin/bash
message="$1"
url=$2

msg_content=\"$message\"

## discord webhook
curl -H "Content-Type: application/json" -X POST -d "{\"content\": $msg_content}" $url