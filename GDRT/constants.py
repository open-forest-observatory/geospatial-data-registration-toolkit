import typing
from pathlib import Path

PATH_TYPE = typing.Union[Path, str]

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
