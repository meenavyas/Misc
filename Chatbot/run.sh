rasa init -v --init-dir /tmp/demo
# Manually Modify domain.yml, data/stories.yml, data/rules.yml and data/nlu.yml
# and store in ~
cp ~/domain.yml /tmp/demo/
cp ~/nlu.yml /tmp/demo/data/
cp ~/stories.yml /tmp/demo/data/
cp ~/rules.yml /tmp/demo/data/

rasa data validate
rasa train --fixed-model-name mybot
rasa shell --model mybot
