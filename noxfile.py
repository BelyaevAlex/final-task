import nox

@nox.session
def black(session):
    session.install("black")
    session.run("black", "src/")

@nox.session
def mypy(session):
    session.install("mypy")
    session.run("mypy", "src/")

@nox.session
def test(session):
    """Run the test suite."""
    session.install("poetry")
    session.install("pytest")
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", external=True)