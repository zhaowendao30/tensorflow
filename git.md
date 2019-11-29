# git 基础命令

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
cat p.md

# 显示每次的命令--在关闭git后查看版本号
git reflog

# 查看工作区和版本库里面最新版本的区别
git diff HEAD -- p1.md

# 丢弃工作区的修改--实质是用版本库中的版本代替工作区的版本
git checkout -- p1.md

# 丢弃暂存区的修改
git reset HEAD p1.md

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

