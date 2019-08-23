#!/usr/bin/expect -f

set timeout 60

set folder_src /home/elie.delplace/M2M_Dogs

set session1 "gpu-server:"
set session2 "elie-Precision-7510:"

set script merge_log.sh

spawn /usr/script_perso/Upload_Nvidia.sh $script

spawn /usr/script_perso/Connect_Nvidia.sh

expect "Last login"
expect "$session1"
send "./$script $folder_src\n"
expect "$session1"
send "exit\n"
expect "$session2"

spawn /usr/script_perso/Download_Nvidia.sh $folder_src/Results $env(PWD)/..

expect "$session2"

spawn /usr/script_perso/Connect_Nvidia.sh

expect "Last login"
expect "$session1"
send "rm -rf $folder_src/Results\n"
expect "$session1"
send "rm -f merge_log.sh\n"
expect "$session1"
send "exit\n"

interact