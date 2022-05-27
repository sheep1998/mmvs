#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:25:49 2022

@author: lihu
"""

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = "google/pegasus-billsum"
device="cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

text = """
         (CNN)For the second time during his papacy, Pope Francis has announced a new group of bishops and archbishops set to become cardinals -- and they come from all over the world.
        Pope Francis said Sunday that he would hold a meeting of cardinals on February 14 "during which I will name 15 new Cardinals who, coming from 13 countries from every continent, manifest the indissoluble links between the Church of Rome and the particular Churches present in the world," according to Vatican Radio.
        New cardinals are always important because they set the tone in the church and also elect the next pope, CNN Senior Vatican Analyst John L. Allen said. They are sometimes referred to as the princes of the Catholic Church.
        The new cardinals come from countries such as Ethiopia, New Zealand and Myanmar.
        "This is a pope who very much wants to reach out to people on the margins, and you clearly see that in this set," Allen said. "You're talking about cardinals from typically overlooked places, like Cape Verde, the Pacific island of Tonga, Panama, Thailand, Uruguay."
        But for the second time since Francis' election, no Americans made the list.
        "Francis' pattern is very clear: He wants to go to the geographical peripheries rather than places that are already top-heavy with cardinals," Allen said.
        Christopher Bellitto, a professor of church history at Kean University in New Jersey, noted that Francis announced his new slate of cardinals on the Catholic Feast of the Epiphany, which commemorates the visit of the Magi to Jesus' birthplace in Bethlehem.
        "On feast of three wise men from far away, the Pope's choices for cardinal say that every local church deserves a place at the big table."
        In other words, Francis wants a more decentralized church and wants to hear reform ideas from small communities that sit far from Catholicism's power centers, Bellitto said.
        That doesn't mean Francis is the first pontiff to appoint cardinals from the developing world, though. Beginning in the 1920s, an increasing number of Latin American churchmen were named cardinals, and in the 1960s, St. John XXIII, whom Francis canonized last year, appointed the first cardinals from Japan, the Philippines and Africa.
        In addition to the 15 new cardinals Francis named on Sunday, five retired archbishops and bishops will also be honored as cardinals.
        Last year, Pope Francis appointed 19 new cardinals, including bishops from Haiti and Burkina Faso.
        CNN's Daniel Burke and Christabelle Fombu contributed to this report.
"""

# inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors='pt')

# outputs = model.model.encoder(**inputs,output_hidden_states=True)

# embeddings = outputs.last_hidden_state

# outputs = model.model.encoder(inputs_embeds = embeddings,output_hidden_states=True)

# print(outputs.last_hidden_state.shape)
# print(len(outputs.hidden_states))

print(model.config)



