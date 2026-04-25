# Unknown Area Navigation

### File Header
Please put a header of this format at the top of each file. This will help with the documentation process and make the final publication stronger.
```python
'''
Author: Ivan Torriani
Description: This file uses the Moondream multimodal model
to convert .jpg images to textual descriptions in json format.
'''
```
### Setup
Basic Setup Instructions, feel free to skip if you've cloned repositories before.

1. On the repository homepage, click the green `Code` button and copy the URL.
2. On your local machine, open a terminal and navigate to where you want the project to live.
3. Type `git clone <url>` with the copied URL to download the repository.
4. Navigate into the project folder with `cd <repo-name>`.


## Branches and Pull Requests
Please use branches and pull requests when completing tasks so that our working code remains intact and our work doesn't become overwritten. 
If you already know how this works, feel free to skip. 

### Branching Setup
1. Next to `main` on the repository homepage, click `Branches`.
2. Make a new branch by clicking `New Branch` with source `main` and give it an legible title with your last name somewhere (ex. Torriani-Autonomous-Team)
3. On your local cloned repo, type in the terminal `git pull` to acknowledge your new branch and `git checkout <your branch>` to select your branch.
4. Whenever you push your code with `git push`, make sure that you are on your branch such that code doesn't get overwritten (I will add safeguards though just in case)

### Pull Requests
1. When you've completed a task that is ready to be incorporated, go to the repository homepage and click `Pull Requests`.
2. Click `New Pull Request` and set the `base` branch to `main` and the `compare` branch to your branch.
3. Give your pull request a clear title and description of what changes you made and select me as a reviewer.
4. Click `Create Pull Request`.

## Commit Message Formalities
Please commit reasonably regularly and use general commit message structure so that it's easier for me to review work and add it in. 
Most importantly, use "feat" when introducing a new feature/new code and "fix" when debugging. For example:
> "feat * added motion detection script"
> "fix * fixed bug in motion detection script"
