# git 教程

## git基础命令

```git
# 初始化文件
git init

每次修改要先add到缓存区，然后才能commit到版本库中去
# 添加文件-可以一次添加多个
git add 完整文件名
eg: git add a.txt b.txt

# 将工作区的所有文件添加到缓存区
git add .

# 提交文件--提交缓存区中的文件
git commit -m '说明'

# 查看命令结果，仓库当前状态
git status

# 查看具体内容
git diff

# 查看历史记录-从近到远显示提交日志
# commit后面的一堆是版本号
git log

# 回退到上一个版本
# 在git中，HEAD表示当前版本， HEAD^表示上一个版本
git reset --hard HEAD^
# 回到指定版本
git reset --hard 版本号(不必写全，git会自己找)

# 查看文本内容
cat <name>

# 显示每次的命令--在关闭git后查看版本号
git reflog

# 查看工作区和版本库里面最新版本的区别
git diff HEAD -- <name>

# 丢弃工作区的修改--实质是用版本库中的版本代替工作区的版本
git checkout -- <name>

# 丢弃暂存区的修改
git reset HEAD <name>

# 删除版本库中的文件
git rm p1.md
git commit -m 'remove'

# 关联github远程库
git remote add origin git@github.com:zhaowendao30/tensorflow.git

# 将本地库中的所有内容推送到远程库上
git push -u origin master

# 整理合并本地库与远程仓库--然后进行上一步
git pull --rebase origin master From github.com:zhaowendao30/tensorflow


# 后期本地提交
git push origin master

# 克隆远程库
git clone git@github.com:zhaowendao30/tensorflow.git
```

## 分支管理

### 创建与合并分支

```git
# 创建分支dev
git branch dev

# 切换分支--有两种方法
1.
git checkout dev

2.
git switch -c dev
git switch master

# 创建并切换分支dev
git checkout -b dev

# 查看当前分支--该命令会列出所有分支，当前分支前面会标一个*
git branch

# 合并分支
git merge dev

# 删除分支
git branch -d dev

# 强删除分支
git branch -D dev
```

### 解决冲突

当不同分支合并冲突时，先解决冲突，然后合并
```git
# 用带参数的git log可以看到分支的合并情况
git log --graph --pretty=oneline --abbrev-commit
```

### 分支管理策略

通常，合并分支时，如果可能，Git会用Fast forward模式，但这种模式下，删除分支后，会丢掉分支信息。

如果要强制禁用Fast forward模式，Git就会在merge时生成一个新的commit，这样，从分支历史上就可以看出分支信息。

```git
# --no-ff表示禁用Fast forword
git merge --no-ff -m 'merge with no-ff' dev
```

### Bug分支

```git
# 将当前的工作现场隐藏起来
git stash

# 查看工作现场
git stash list

# 恢复工作现场
1.恢复工作现场，但是stash的内容并不删除
git stash apply
删除stash内容
git stash drop

2.恢复工作现场并删除stash内容
git stash pop

# 恢复指定的stash内容
git stash apply stash@{0}

# 讲bug提交的修改复制到分支上
git cherry-pick <commit>
```

### 多人协作

```git 
# 查看远程库
git remote
# 查看远程库的详细信息
git remote -v

# 从本地推送分支
git push origin branch-name
# 若推送失败--有人先我们一步推送了分支，使用git pull抓取远程的新提交
git pull

# 在本地创建和远程对应的分支
git checkout -b branch-name origin/branch-name

# 建立本地分支和远程分支的关联
git branch --set-upstream branch-name origin/branch-name
```


rebase
1.rebase操作可以吧本地未push的分叉提交历史整理成直线
2.rebase的目的是使得我们在查看历史提交的变化时更容易，因为分叉的提交需要三方对比
```git
git rebase
```


## 标签管理

### 创建标签

**标签不是按时间顺序列出，而是按字母排序**
```git
# 首先切换到需要打标签的分支上
# 创建标签
git tag <name>

# 查看所有标签
git tag

# 对具体的提交打标签
git tag <标签> <commit>
# 对commit id 为 f52c633设置标签为v0.9
git tag v0.9 f52c633

# 查看标签信息
git show <name>

# 创建带有说明的标签， 用-a制定标签名, -m制定说明文字
git tag -a v0.1 -m 'version 0.1 released' <commit>
```

### 操作标签

```git
# 删除标签
git tag -d <tagname>

# 推送某个标签到远程
git push origin <tagname>

# 推送所有标签到远程
git push origin --tags

# 删除远程标签
1.删除本地标签
git tag -d <name>
2.删除远程标签
git push origin :refs/tags/<tagname>
```

## 使用Github

* 在Github上，可以任意Fork开元仓库
* 自己拥有Fork后的仓库的读写权限
* 可以推送pull request给官方仓库来贡献代码


### 关联多个仓库
```git
# 不能使用origin
# 关联github仓库
git remote add github git@github.com:zhaowendao30/tensorflow.git
# 关联码云的仓库
git remote add gitee git@gitee.com:zhaowendao30/tensorflow.git

# 推送到github
git push github master
# 推送到码云
git push gitee master
```


## 自定义git

```git
# 让Git显示颜色
git config --global color.ui true
```


### 忽略特殊文件
原则：
* 忽略操作系统自动生成的文件，比如缩略图等
* 忽略编译生成的中间文件、可执行文件等，也就是如果一个文件是通过另一个文件自动生成的，那自动生成的文件就没必要放进版本库，比如Java编译产生的.class文件
* 忽略你自己的带有敏感信息的配置文件，比如存放口令的配置文件。
在git工作目录下创建一个.gitignore文件，将要忽略的文件名填进去，然后将.gitignore提交

```git
# 添加被.gitignore忽略的文件
git add -f <name>

# 查看.gitignore规则
git check-ignore -v <name>
```


### 配置别名

```git
# 将status设置别名st
git config --global alias st status

# 配置unstage代替reset HEAD
git config --global alias. unstage 'reset HEAD'

# 配置git last显示最近一次提交
git config --global alias. last 'log -1'

# 配置lg--不知道是啥，但是好像很强--先记下来
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```
* 配置Git的时候，加上--global是针对当前用户起作用的，如果不加，那只针对当前的仓库起作用。

* 每个仓库的Git配置文件都放在.git/config文件中

* 别名就在[alias]后面，要删除别名，直接把对应的行删掉即可，配置别名也可以直接修改这个文件，如果改错了，可以删掉文件重新通过命令配置



### 搭建git服务器--目前来说没啥用--先不学了

### Source Tree -- git [图形工具](https://www.liaoxuefeng.com/wiki/896043488029600/1317161920364578)

