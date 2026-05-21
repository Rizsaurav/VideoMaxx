from abc import ABC, abstractmethod

from vidmaxx.models.asset import AssetCandidate


class AssetSourceBase(ABC):
    @abstractmethod
    async def search(self, query: str, n: int, images_only: bool = False) -> list[AssetCandidate]:
        """Return up to n candidates for the given query."""
        ...
