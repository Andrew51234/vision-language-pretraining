# Vision Language Pretraining



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/tsch.maulikk/vlp.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.lrz.de/tsch.maulikk/vlp/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
**[NOTE] This is a public clone of a private repository (https://github.com/tjdevWorks/ConVIRT-Federated) and it may not be up-to-date with the latest changes.**
______________________________________________________________________

<div align="center">

#  Exploring the Potential of Federated Learning for Medical Image Analysis in Non-IID Settings

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

This repository contains code to train the ConVIRT model on the MIMIC-CXR-JPG dataset and fine tune the pretrained image backbone for downstream image multi-label classification on the CheXpert dataset in centralized and federated learning setups.
<br><br>

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ag2307/ConVIRT-Federated
cd ConVIRT-Federated

# [OPTIONAL] create conda environment
conda create -n convirt_fed python=3.7
conda activate convirt_fed

# install requirements
pip install -r requirements.txt
```

Pretraining the model with default configuration

```bash
python src/pretrain.py
```

An example of fine tuning in centralized setup:

```bash
# To use the ConVIRT pretrained model Image Backbone
python src/finetune_chexpert.py

# To use the ImageNet pretrained model image backbone
python src/finetune_chexpert.py --config-name=finetune_chexpert_imagenet
```

To execute the federated learning setups we have three data partitioning strategies in [configs/partitions/](configs/partitions/) volume, class, attribute.

An example of running a federated learning experiment:

```bash
# Runs a federated simulation on a single node with gpu using 4 clients for 100 rounds and paritioning logic for "class.yaml"
python src/run_simulation.py --config-name=prod_simulation server_config.num_rounds=100 pool_size=4 partitions=class partitions.num_clients=4 partitions.exclusive=False partitions.equal_num_samples=False task_name='fed_chexpert_class' job_name=fed_class_100_4_False_False datamodule.batch_size=256
```

You can override any parameter from command line like this

```bash
python src/finetune_chexpert.py trainer.max_epochs=20 datamodule.batch_size=64
```
## Citation
```bash
@article{DBLP:journals/corr/abs-2010-00747,
  author    = {Yuhao Zhang and
               Hang Jiang and
               Yasuhide Miura and
               Christopher D. Manning and
               Curtis P. Langlotz},
  title     = {Contrastive Learning of Medical Visual Representations from Paired
               Images and Text},
  journal   = {CoRR},
  volume    = {abs/2010.00747},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.00747},
  eprinttype = {arXiv},
  eprint    = {2010.00747},
  timestamp = {Fri, 20 Nov 2020 14:04:05 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-00747.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Contributors

Tejas Mahajan

Sai Charitha Akula

Abhinav Gupta
