# ИМЯ ОБРАЗА: МОЖНО МЕНЯТЬ
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
# МЕНЯТЬ НЕ НАДО
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG='C.UTF-8'
ARG USERNAME=vscode
ENV USERNAME=${USERNAME}    

# МЕНЯТЬ НЕ НАДО; УСТАНОВКА SUDO
RUN apt-get update && apt-get install -y --no-install-recommends sudo && \
    apt-get install -y --no-install-recommends cmake rapidjson-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# ВАЖНО ОСТАВИТЬ
RUN useradd -m -d /workspaces -s /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL >> /etc/sudoers
# RUN sudo apt-get update && sudo apt-get install curl && curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash && sudo apt-get install -y nodejs
WORKDIR /workspaces
RUN apt update && apt install git -y
COPY tbank_imagegen/requirements.txt .
RUN pip install -r requirements.txt

USER ${USERNAME}

CMD ["python", "/workspaces/tbank_imagegen/InstantCharacter/final_main.py"]
