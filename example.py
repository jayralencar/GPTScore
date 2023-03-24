from flan_score import FLANScorer

src = ["A young woman who was subjected to years of abuse by Bega Cheese boss Maurice Van Ryn has told a court of the severe impact it had on her life. For years she kept what Van Ryn did to her a secret, but when she found out he had been caught she knew she had to break her silence. The wealthy NSW businessman has pleaded guilty to 12 sex offences, including the most serious charge of persistent sexual abuse of a child. One of his victims told Sydney's District Court on Monday that she is scared for her future relationships, and didn't understand why she had been targeted. Scroll down for video . A young woman who was subjected to years of abuse by Bega Cheese boss Maurice Van Ryn (pictured) has told a court of the severe impact it had on her life . The wealthy NSW businessman has pleaded guilty to 12 sex offences, including the most serious charge of persistent sexual abuse of a child . 'I am scared for my future relationships,' the young woman said, according to News Corp. 'He has stolen my right to share my body with someone for the first time.' Van Ryn sat calmly in the dock and did not look at the young woman as she told Sydney's District Court on Monday about the impact his abuse has had on her. The young woman, who cannot be named, said when he abused her as a child she didn't know it was wrong. 'I didn't understand why he would do something like this. Does this happen to every kid? Do friends do this to other friends?' she said. 'I didn't know it was against the law.' Van Ryn (left) sat calmly in the dock and did not look at the young woman as she told Sydney's District Court on Monday about the impact his abuse has had on her. The young woman, who cannot be named, said when he abused her as a child she didn't know it was wrong . She said she hadn't spoken up about the abuse until the day she heard he had been caught, and since then had attempted to take her own life. 'I knew I had to say something.' Since then, however, she says she has been scared people will find out. 'I didn't want people to know what happened to me. I don't want people asking questions. She says she wonders when Van Ryn, who is now in custody will be out in the community again, and she worries what he did to her will forever affect how she relates to men. She said she has suffered depression since she was 14 and had been hospitalised for self harm and attempted suicide. The hearing continues. Readers seeking support and information about suicide prevention can contact Lifeline on 13 11 14. Sorry we are not currently accepting comments on this article.\ntl;dr:\n"]*2
tgt = [
    "The Crown is a historical drama web television series created and written by Peter Morgan and produced by Left Bank Pictures and Sony Pictures Television for Netflix. It is a biographical story about the reign of Queen Elizabeth II of the United Kingdom. The first season covers the period from her marriage to Philip, Duke of Edinburgh in 1947 to the disintegration of her sister Princess Margaret’s engagement to Peter Townsend in 1955. The second season covers the period from the Suez Crisis in 1956 through the retirement of the Queen’s third Prime Minister, Harold Macmillan, in 1963 to the birth of Prince Edward in 1964. The third season will continue from 1964, covering Harold Wilson’s two terms as the Prime Minister until 1976, while the fourth will see Margaret Thatcher’s premiership and a focus on Diana, Princess of Wales. The series is intended to last 60 episodes over six seasons, with 10 one-hour episodes per season, and new actors being cast every two seasons. The first season was released on Netflix on November 4, 2016, with the second released on December 8, 2017.",
    "a young woman who was subjected to years of abuse by bega cheese boss maurice van ryn has told a court of the severe impact it had on her life . for years she kept what van ryn did to her a secret, but when she found out he had been caught she knew she had to break her silence . the wealthy nsw businessman has pleaded guilty to 12 sex offences, including the most serious charge of persistent sexual abuse of a child ."
]

template = "Rewrite the following text with consistent facts:\n{0}\nIn other words, "
src = [template.format(d) for d in src]

scorer = FLANScorer(device="cpu",checkpoint="google/flan-t5-base")

sc = scorer.score(src, tgt,2,weighted=False)

print(sc)