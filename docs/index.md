# Welcome

Welcome to my example website! This website uses
[MkDocs](https://www.mkdocs.org/) with the [Material
theme](https://squidfunk.github.io/mkdocs-material/) and an automated
deployment workflow for publishing to [GitHub
Pages](https://pages.github.com/).

This guide will help you create your own GitHub Pages website with
this setup. If you're using Windows, you should run all the commands
in this guide on a Windows Subsystem for Linux (WSL) terminal.

## Getting Started

### Initializing your Website Repository

First, [create a new repository](https://github.com/new) on
GitHub. Make sure to skip the initialization step on the website
&mdash; you will be initializing the contents of this repository with
this template! Note down the Git remote for your new repository,
you'll need it later when initializing your local copy of the
repository.

Next, download the website template and extract it:

```sh
$ wget https://github.com/jansky/test-website/archive/refs/tags/template.tar.gz
$ tar xvf template.tar.gz
$ rm template.tar.gz
```

!!! note

    If the `wget` command is not found, you can install it using
    apt-get: `sudo apt-get install wget`.

This will create a new folder called `test-website-template` in
your current directory with the website template contents. You may
wish to rename this folder to something like `my-website`:

```sh
$ mv test-website-template my-website
```

Now you can initialize a Git repository with the website contents:

```sh
$ cd my-website
$ git init
$ git remote add origin YOUR_GITHUB_REPOSITORY_REMOTE
```

### Website Configuration

The configuration for your website is stored in `mkdocs.yml` in the
repository root. You only need to change a few settings at the
top of the file:

```yaml linenums="1"
site_name: Example Website
site_url: https://reighns92.github.io/test-website
nav:
  - Home: index.md
  - About: about.md
  - Notebooks: notebooks_list.md
...
```

First, update the `site_name` and `site_url` fields to be correct for
your website. The URL format for GitHub pages websites is

```
https://USERNAME.github.io/REPOSITORY-NAME
```

As you add content to your website, you can also control the pages
that appear on your website's navbar in the `nav` field. Each `nav`
list element is of the form

```yaml
Link Text: filename.md
```

For navbar links pointing to pages in your site, you should use a file
path which is relative to the `docs/` folder where all your website
content is stored. You may also link to external pages and include
sub-links. For more information, you can view the [MkDocs nav
documentation](https://www.mkdocs.org/user-guide/writing-your-docs/#configure-pages-and-navigation).

### GitHub Actions Configuration

You also need to update the GitHub Actions deployment workflow
with the name and e-mail address to use when the workflow
pushes your built website to the `gh-pages` branch of your
repository. In the file `.github/workflows/deploy-website.yml`,
update lines 25 and 26 to reflect your account information:

```yaml linenums="22" hl_lines="4 5"
...
- name: Push Build Website to gh-pages Branch
  run: |
   git config --global user.name 'YOUR NAME(Automated)'
   git config --global user.email 'YOUR-GITHUB-USERNAME@users.noreply.github.com'
...
```

### Setting Up Local Development

MkDocs makes it easy to develop your website locally and see your
changes in real time. To begin, set up and activate Python virtual
environment for this project. Then, install the project dependencies:

```sh
(venv) $ pip install -r requirements.txt
```

MkDocs includes a small webserver that allows you to preview your
website in your browser as you make changes. Whenever you save one of
your source files, the website will be rebuilt and your browser will
automatically refresh to show the new changes. You can start this
development server using the `serve` command:

```sh hl_lines="5"
(venv) $ mkdocs serve
INFO     -  Building documentation...
...
INFO     -  Documentation built in 0.16 seconds
INFO     -  [20:09:07] Serving on http://127.0.0.1:8000/...
...
```

If you copy and paste the URL given by MkDocs into your browser you
will see your website preview.

### Adding Website Content

Markdown files added under the `docs/` folder will be converted
to HTML pages on your website. This website template enables some
helpful extensions for

- rendering of math in LaTeX style,
- admonitions such as notes and warnings, and
- code blocks with syntax highlighting.

In addition, MkDocs also supports GitHub-flavored Markdown tables. To
see examples of syntax for these elements, see
[example.md](example.md).

If you add files of a type other than Markdown (e.g., images, IPython
Notebooks) into the `docs/` folder, these will be copied over to your
built website. See for example the [Notebooks](notebooks_list.md) section
of this template which provides links to download IPython notebooks in
this repository.

### Deploying Your Changes

When you are ready to deploy your website for the first time, make an
initial commit and push to your GitHub remote:

```sh
$ git add .
$ git commit -a -m "Initial Commit"
$ git push origin master -u
```

The GitHub Actions deployment workflow included with this template
runs whenever you push to the `master` branch. This workflow will
build your website using MkDocs and push the built website files to
the `gh-pages` branch of your repository, which GitHub Pages can use
to serve your website.

Once you have pushed your changes, go to your repository page on
GitHub and confirm that the GitHub Actions workflow has completed
successfully (you should see a green checkmark next to the name of the
most recent commit). Then, go to your repository settings page, and
click on 'Pages'. You will see a section that will let you set the
source for your GitHub Pages website. Click the box labelled 'None'
and select the `gh-pages` branch. Leave the selectd folder at '/
(root)' and click 'Save'. Your website is now live!

To push additional changes, simply commit and push to the master
branch. The GitHub Actions deployment workflow will handle deploying
your chnages to GitHub Pages.





