---
title: Tmux 会话(Session)/窗口(Window) 重命名快捷键汇总
date: 2021-06-16 18:28:44
tags:
---

## 会话（Session）重命名

使用快捷键重命名：

`Ctrl + x, $`

使用指令重命名，按快捷键进入指令模式

`Ctrl + x, :`

输入下述指令，指定当前会话名称和新的会话名称

`rename-session [-t current-name] [new-name]`

## 窗口（Windows）重命名

使用快捷键来重命名

`Ctrl + x, ,`

使用指令重命名，类似上面提到的进入命令模式

`Ctrl + x, :`

输入窗口重命名指令：

`rename-window [-t current-name] [new-name]`

