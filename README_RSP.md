# Contribution instructions from an RSP

This is a set of instructions for how to get a source checkout of hyrax working on an RSP. You will need to do this if you are developing a hyrax feature or modifying one of the in-built dataset classes.

The instructions are tailored to the usdf RSP but should be modifiable for other RSPs

## Setting up your repository
First add your SSH key to github. This is so you will eventually be able to push your changes.
on USDF the public version of you key is in `~/.ssh/s3df/id_ed25519.pub`. You will need to paste the contents of this key into github [here](https://github.com/settings/ssh/new) to add it to your account.

Next you need to tell git to use this key when it uses ssh. This is not working by default on USDF, so you will need to clone the repository with the command below. It's recommended you run this from your `~/rubin-user` directory such that your checkout will be in `~/rubin-user/hyrax`. This way it will persist between notebook server invocations in USDF.

```
GIT_SSH_COMMAND='ssh -i ~/.ssh/s3df/id_ed25519 -o IdentitiesOnly=yes' git clone git@github.com:lincc-frameworks/hyrax.git
```

After this you an enter the directory and run the following command to make sure all future git commands in this repository will use your private key:

```
git config --global core.sshCommand 'ssh -i ~/.ssh/s3df/id_ed25519 -o IdentitiesOnly=yes'
```

At this point you have an active git repository and you should be able to `git fetch` without error. It is recommended that you set your global username and email with the following commands if you intend to commit with the following commands:

```
git config --global user.email "email@domain.tld"
git config --global user.name "Your Name"
```

## Setting up your notebook
After this setup you can run the following notebook magic to bootstrap an editable install of hyrax in your notebook environment

```
# In the first cell of your notebook
%pip install -q -e ~/rubin-user/hyrax 2>&1 | grep -vE 'WARNING: Error parsing dependencies of (lsst-|astshim|astro-)'
```

This command may suggest you restart your kernel, and you should do so if it asks. When you edit hyrax you will need to restart your kernel, but will not need to re-run this install command.

## Running Hyrax

You can now run hyrax in a notebook cell. This is a sample that uses the LSSTDataset class, which only functions inside of the RSP

```
import hyrax
h = hyrax.Hyrax()
h.config["data_set"]["name"] = "LSSTDataset"

d = h.prepare()

d[0].shape
```

