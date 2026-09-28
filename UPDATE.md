# How to apply this update

Unpack over the repository root, keeping your `results/` and `figures/`
untouched. The archive contains no results, so nothing of yours is overwritten.

```bash
cd sample-efficiency-mujoco
unzip -o ~/Downloads/sample-efficiency-mujoco_update-01.zip
```

First time only, build the commit history:

```bash
bash setup_commits.sh
git log --oneline
```

If your repository already has a single commit containing everything and you
want the staged history instead, discard it first. Nothing is lost, the files
on disk are untouched:

```bash
rm -rf .git
bash setup_commits.sh
```

On later updates, unpack again and run the same script. Steps already in the
history are skipped, and anything new is listed for you to commit yourself:

```bash
unzip -o ~/Downloads/sample-efficiency-mujoco_update-NN.zip
bash setup_commits.sh
git add -A && git commit -m "<message supplied with the update>"
```
