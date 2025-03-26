# Influencer Games 
The main focus of this package is to provide a framework to studying the dynmaics of spatial influence in a multi-player resource compeition game. The package takes two paths to study influence games
1. Adaptive dynamics (gradient ascent of the reward vectors)
    -By following the gradient of the rewards avaiable to a player they can move toward a maximia value for thier return
2. Multi Agent Reinforcement learning 
    - Q-learning: Independent Q-learning
    - Deep reinforcement learning from [RAYRL](https://docs.ray.io/en/latest/rllib/index.html)

The project is based on the work of Mark Lovett, Feng Fu, and Alex McAvoy at Dartmouth Mathematics. 
    
- [Paper](https://www.google.com/): Original paper with theoretical results.
## An influence game form 
The payoff for player $i$ ($u_i(\mathbf{x})$) of an influence game over a resource distribution $B=\{B(b)|b\in \mathbb{B}\}$ is the expected return of resources for a player. Given a discrete resource distribution $u_i$ has the following form
$$
    u_i(\mathbf{x})=\sum_{b\in \mathbb{B}} B(b)G_{i}(x_i,x_{-i},b).
$$
For a continous resource distribution the game has the following form 
$$
    u_i(\mathbf{x})=int_{\mathbb{B}} B(b)G_{i}(x_i,x_{-i},b)db.
$$
Where $G_{i}(x_i,x_{-i},b)$ is the probability of influence and is the ratio of a players influence over a resource point to the sum of all influence on that point. The proability of influence is defined as 
$$ 
    G_i(x_i,x_{-i},b)= \frac{f_{i}(x_i,b)}{\sum_{j\in I} f_{j}(x_j,b)},
$$
where $f_{i}(x_i,b)$ is the $i$th players influence over a resource point $b$.

## Why study influencer games?
There are many games in society that take the form of a influence game with spatial influence playing a major role. To list a few:
1. Voter dynamics: The compition for votes or influence over the masses in poltics is an exmaple where polticians are players who must optimize thier expected return of votes to win elections. politcians can optimize thier voter turn by changing thier ideological point of view or that of thier campaign at least or by changing the reach of thier influence to reach un-touched voter populations who choese an "abstaining" canidate. 
2. Fishing and net choice: Fisherman try to optimize thier hual via choosing optimal finishing territories aka thier influence over fish populations or by chainging thier strategies to harvest thier territory more efficently (effectively chaning thie reach)
3. Marketing and content genration: Influencers and marketers compete for limited consumer view time by choosing content topics in the form of a zero sum influence game
4. Market maker compeition: Market makers comepete for resources in the form of volumes of trades, which creates a zero sum influence game.
5. And many more  

The influce game package allows scholars to use a code frame work to study these games and understand how thier influence kernels impact the existance and stability of symmetric Nash. 