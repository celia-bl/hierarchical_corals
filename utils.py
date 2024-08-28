class CaseInsensitiveDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convertir toutes les clés existantes en minuscules
        self._convert_keys()

    def _convert_keys(self):
        """Convertit toutes les clés du dictionnaire en minuscules."""
        for key in list(self.keys()):
            value = super().pop(key)
            super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())

