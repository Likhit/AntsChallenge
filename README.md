# Ants Challenge

This repo modifies the Fall 2011 [AI Challenge](https://github.com/aichallenge/aichallenge). The repo only contains code needed to run the ants game (and hence the website and competition code has been removed).

## Folder Contents

* `ants/` - Everything related to ants: engine, starter packages, maps/mapgen, visualizer
* `worker/` - Standalone workers which run games (including compiler and sandbox)
* `bots/` - Code for existing bots (including starter bots, example bots, and some competition winner bots)

## Competition bots

Some bots which competed in the Fall 2011 competition have been added as submodules in the `bots\winner_bots` folder. To initialize the git submodules run:

* `git submodule init`
* `git submodule update`

Explanation for these bots:

* [xathis - First Place](https://web.archive.org/web/20120215072119/http://xathis.com/posts/ai-challenge-2011-ants.html)
* [a1k0n - Eleventh Place](https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html)
