# Releasing

Publishing is performed only when a `v*` tag is pushed. Do not create a release
tag until the release PR has merged and its checks have passed.

## One-Time Setup

1. Create a pending Trusted Publisher at
   <https://pypi.org/manage/account/publishing/> for `linz-s3-utils`.
2. Set the owner to `quinnhornblow`, repository to `linz-s3-utils`, workflow to
   `.github/workflows/publish.yml`, and environment to `pypi`.
3. Create the `pypi` GitHub environment and require manual approval before
   deployment.

## Release Steps

1. Confirm CI passes on the exact `main` commit to be released.
2. Run the live integration suite once:

   ```bash
   uv run pytest --run-integration
   ```

3. Create and push a matching version tag, for example `v0.1.0`.
4. Approve the `pypi` deployment when GitHub Actions requests it.
5. Verify installation from PyPI:

   ```bash
   python -m pip install linz-s3-utils
   ```

6. Create GitHub release notes and close the release-readiness issue.
