<div align=center>
  <h1>MindSpore Compatible Distributed Training: Principles and Practices</h1>
  <p><a href="./README_ZH.md">查看中文</a></p>
</div>

This course introduces the end-to-end development process based on the MindSpeed-Core-MS suite, focusing on scenarios 
such as LLM pre-training, fine-tuning, and reinforcement learning. 
It enables learners to understand the related concepts, features, and development workflows of MindSpeed-Core-MS, 
equipping them with foundational capabilities for large model training across different scenarios.
## 📢 News

- 2025-11-30 「Course Update」: Added chapters 1-5, including complete videos, courseware, and code examples. [View Details](xxxx)

[//]: # (- 2025-10-18 「Feature Optimization」: Project repository refactored, course resources easier to find, new PR)

[//]: # (    check gates added for more standardized content integration. [View Details]&#40;xxx&#41;)

[//]: # (- 2025-10-10 「Bug Fix」: Fixed xxxxxx issue, thanks to @username for PR contribution. [View Details]&#40;xxx&#41;)


## Prerequisites

This course is an intermediate-level course in MindSpore course series. It is recommended that readers have knowledge of Transformer architecture and single-machine model training before taking this course.


## Environment Setup

The Docker environment used in this course can be obtained from [dockerfiles](./dockerfiles/)
Main dependencies are as follows:

| Dependency                                                                                    | Version   |
|:----------------------------------------------------------------------------------------------|:-------|
| [CANN](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1) | 8.2RC1 |
| Python                                                                                        | \>=3.9 |
| [MindSpeed-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/tree/r0.4.0)                 | r0.4.0 |


## Course Content

| No. | Lesson                                                                                                            | Description                                                                                           | Learning Resource                                              |
|:---|:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|:--------------------------------------------------|
| 1  | Overview of MindSpeed-Core-MS: MindSpore-Compatible LLM Training Suite                                            | Introduces the architecture and capabilities of the MindSpeed-Core-MS                                 | [PPT](./Chapter1) · [手册](./Chapter1) · [视频](跳转链接) |
| 2  | Pre-training Practice Based on MindSpeed-Core-MS                                                                  | Covers the process and practice of large model pre-training using MindSpeed-Core-MS                   | [PPT](./Chapter2) · [手册](./Chapter2) · [视频](跳转链接) |
| 3  | Principle and Practice of Fine-tuning Based on MindSpeed-Core-MS                                                  | Explores the process and practice of fine-tuning using MindSpeed-Core-MS                              | [PPT](./Chapter3) · [手册](./Chapter3) · [视频](跳转链接) |
| 4  | Principle and Practice of Reinforcement Learning Based on MindSpeed-Core-MS                                                                                    | Discusses the process and practice of reinforcement learning using MindSpeed-Core-MS                  | [PPT](./Chapter4) · [手册](./Chapter4) · [视频](跳转链接) |
| 5  | Introduction and Practice of Memory & Performance Tuning                                                                                                    | Introduces the methodologies and practical applications of memory optimization and performance tuning | [PPT](./Chapter5) · [手册](./Chapter5) · [视频](跳转链接) |

## Version Management

This repository is updated in sync with [MindSpore](https://www.mindspore.cn/install) and [MindSpeed-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/tree/master), New releases of this repository are published approximately every **six months**.

| Branch/Version  | Python | MindSpore | MindSpeed-Core-MS |
| :----- |:-------|:----------|:------------------|
| master | \>=3.9 | 2.7.1     | r0.4.0            |


## FAQ

See the [FAQ](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/FAQ) in the Wiki。

## Contributing

We welcome bug reports, suggestions, and code contributions via [Issue](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/issues) or [PR](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/pulls)
. Please follow our submission guidelines — all PRs are reviewed and merged by @username. Your contributions make the project stronger!

**Guidelines:** [Issue & PR Submission](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/%E6%8F%90%E4%BA%A4%E8%AF%B4%E6%98%8E)

### Contributors

Special thanks to all contributors for improving this project!

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices" />
  </a>
</div>
