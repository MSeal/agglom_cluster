Update the CHANGELOG.md with any significant changes for the release.

Then Ensure that you've build the extension for the release.

   pip install -r requirements-dev.txt
   python setup.py build_ext

You should now see a new cluster.so in the hac directory.

   git add hac/cluster.so
   git commit -m "Compiled new .so for release"

To release, run through the following:

    rm -rf dist
    # Update VERSION file with <next version>
    git tag <next version>
    python setup.py sdist
    # Check the tar version matches expected release version
    git push --tags
    python setup.py sdist upload
