To release, run through the following:

    rm -rf dist
    # Update VERSION file with <next version>
    git tag <next version>
    python setup.py sdist
    # Check the tar version matches expected release version
    git push --tags
    python setup.py upload
