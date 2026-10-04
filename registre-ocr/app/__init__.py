"""Couche application : agent conversationnel type WhatsApp, file hors ligne chiffrée, liaison patiente.

Le pipeline d'extraction (`registre/`) ne sait rien de cette couche : il reçoit des photos et rend un
`Dossier`. Ici on gère ce qui l'entoure sur le téléphone de la sage-femme : capture sans réseau,
cycle de vie de l'enregistrement, révision, rattachement à la patiente, synchronisation.
"""
