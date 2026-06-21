"""50 diverse pretraining-like passages for the steganography grid.

Spread across domains/registers (history, science, law, literature, news, technical,
biology, economics, philosophy, geography, medicine, math, music, sport, cooking, ...)
so that extracted activations span a range of reconstruction quality (the figure's
FVE-norm x-axis). One Joseon-dynasty record (index 0) echoes the paper's Common Pile
running example. Short, raw-ish prose — no instructions/questions.
"""

PROMPTS = [
    # history
    "In the third year of King Jungjong's reign, the Veritable Records of the Joseon dynasty noted a severe drought across the southern provinces, prompting the court to perform rain rituals.",
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and entrenched the principle that sovereign states should not interfere in one another's internal affairs.",
    "Hannibal led his army, including a contingent of war elephants, across the Alps in 218 BC to strike at Rome from the north.",
    "During the Meiji Restoration, Japan dismantled the feudal domains and rapidly industrialised, importing foreign engineers to build railways and arsenals.",
    "The Library of Alexandria amassed hundreds of thousands of scrolls before a series of fires and budget cuts gradually eroded its collection.",
    # physical science
    "Mitochondria generate ATP through oxidative phosphorylation across the inner membrane, coupling electron transport to a proton gradient.",
    "A black hole's event horizon marks the boundary beyond which the escape velocity exceeds the speed of light, so no signal can return to a distant observer.",
    "When supercooled water is disturbed, it can crystallise almost instantly as the metastable liquid releases its latent heat of fusion.",
    "Superconductors expel magnetic fields below a critical temperature, a phenomenon known as the Meissner effect that enables frictionless magnetic levitation.",
    "The half-life of carbon-14 is about 5,730 years, which allows archaeologists to date organic remains up to roughly fifty thousand years old.",
    # law / diplomacy
    "The treaty established a demilitarized zone along the river and obligated both signatories to submit their disputes to international arbitration.",
    "Under the doctrine of adverse possession, a trespasser may acquire legal title to land if their occupation is open, continuous, and hostile for the statutory period.",
    "The appellate court remanded the case, holding that the trial judge had improperly excluded exculpatory evidence during the suppression hearing.",
    # literature / narrative
    "The lighthouse keeper climbed the spiral stairs each dusk, lighting the lamp that had guided fishermen home for forty years.",
    "She folded the letter twice, pressed it beneath the loose floorboard, and resolved never to speak of the inheritance again.",
    "The old market town woke slowly: shutters creaked open, a baker swept flour from his step, and pigeons wheeled above the empty square.",
    "He had always believed the river was patient, that it would carry away every grievance if you only waited long enough on its bank.",
    # news / journalism
    "Central bankers signalled on Thursday that interest rates would remain elevated until inflation showed a durable return toward the two percent target.",
    "Rescue crews worked through the night after the embankment collapsed, evacuating several hundred residents from the flooded lowland districts.",
    "The company's quarterly earnings beat analyst expectations, sending its shares up nearly nine percent in after-hours trading.",
    # computing / technical
    "A hash table offers average constant-time lookups by mapping keys to buckets, though collisions can degrade performance to linear in the worst case.",
    "The compiler performs dead-code elimination after constant propagation, removing branches that can never execute given the inferred value ranges.",
    "TCP guarantees ordered delivery by numbering segments and retransmitting any that go unacknowledged before a timeout expires.",
    "Gradient descent updates each parameter in the direction that most steeply reduces the loss, scaled by a learning rate that must be tuned carefully.",
    # biology / medicine
    "The pancreas secretes insulin in response to rising blood glucose, prompting cells throughout the body to take up sugar from the bloodstream.",
    "CRISPR-Cas9 uses a guide RNA to direct a nuclease to a specific genomic sequence, where it introduces a double-strand break for editing.",
    "Antibiotic resistance spreads when bacteria exchange plasmids carrying genes that inactivate or pump out the offending drug.",
    "The immune system's memory cells persist for years after an infection, enabling a faster and stronger response upon re-exposure to the same pathogen.",
    # economics / finance
    "When a currency depreciates, exports become cheaper abroad while imported goods grow more expensive for domestic consumers.",
    "Compound interest causes a balance to grow geometrically, so even modest annual returns accumulate substantially over several decades.",
    "A liquidity trap arises when interest rates approach zero and monetary policy loses its power to stimulate additional borrowing.",
    # philosophy / social science
    "Utilitarian ethics judges an action by the aggregate welfare it produces, which critics argue can license sacrificing the few for the many.",
    "The tragedy of the commons describes how individuals acting in self-interest can deplete a shared resource that none of them wishes to destroy.",
    "Confirmation bias leads people to seek and remember evidence that supports their existing beliefs while discounting what contradicts them.",
    # geography / earth science
    "The Atacama Desert is among the driest places on Earth, with some weather stations never having recorded a single drop of rain.",
    "Plate tectonics drives the slow drift of continents, building mountain ranges where plates collide and opening ocean basins where they separate.",
    "Monsoon winds reverse direction with the seasons, drenching the subcontinent in summer and leaving it parched through the winter months.",
    # mathematics
    "A prime number has exactly two divisors, and Euclid proved more than two thousand years ago that the primes never run out.",
    "The Fibonacci sequence, in which each term is the sum of the two before it, approaches the golden ratio as the terms grow large.",
    "A function is continuous at a point if small changes in its input produce arbitrarily small changes in its output near that point.",
    # arts / music
    "A fugue develops a single short theme by introducing it in one voice and then weaving it through the others at staggered intervals.",
    "Impressionist painters abandoned crisp outlines in favour of loose brushwork that captured the fleeting effects of light on a scene.",
    # sport
    "With two minutes remaining, the midfielder threaded a pass through the defence, and the striker volleyed it into the top corner.",
    "The marathon's final miles punish the unprepared, as depleted glycogen forces runners to slow against what they call the wall.",
    # cooking / everyday
    "To temper chocolate, you gently heat it, cool it while stirring, and warm it again so the cocoa butter sets into a glossy, stable crystal form.",
    "Bread rises because yeast ferments the dough's sugars, releasing carbon dioxide that is trapped by the elastic gluten network.",
    # astronomy / space
    "A comet's tail always points away from the Sun, pushed outward by the solar wind regardless of the direction the comet is travelling.",
    "The James Webb telescope observes in the infrared, peering through dust clouds to capture light from the universe's earliest galaxies.",
    # linguistics / misc
    "Languages in close contact often borrow vocabulary, so a single everyday word may carry traces of three or four older tongues.",
    "Tidal forces gradually slow the Earth's rotation, lengthening the day by a tiny fraction of a second each century.",
]

assert len(PROMPTS) == 50, f"expected 50 prompts, got {len(PROMPTS)}"
